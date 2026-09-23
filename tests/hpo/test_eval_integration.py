import json
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest
import xarray as xr

import chap_core.assessment.evaluation as evaluation_module
from chap_core.api_types import BacktestParams
from chap_core.assessment.evaluation import Evaluation, FlatEvaluationData
from chap_core.assessment.flat_representations import FlatForecasts, FlatObserved
from chap_core.assessment.weather_providers import DEFAULT_WEATHER_PROVIDER_ID
from chap_core.database.model_templates_and_config_tables import ModelConfiguration
from chap_core.hpo.hyperparameter_optimizer import HyperparameterOptimizer
from chap_core.hpo.searcher import GridSearcher
from chap_core.hpo.types import FlatHyperparameterOptimization, HyperparameterOptimization, Trial


def make_hpo_result() -> HyperparameterOptimization:
    return HyperparameterOptimization(
        searcher="GridSearcher",
        model_template_name="demo-template",
        model_template_version="1.0",
        backtest_params=BacktestParams(n_periods=1, n_splits=1, stride=1),
        metric="rmse",
        search_space={"x": [1, 2]},
        max_trials=None,
        seed=17,
        model_configuration=ModelConfiguration(
            user_option_values={"x": 1},
            additional_continuous_covariates=["rainfall"],
        ),
        best_params={"x": 1},
        best_score=1.5,
        leaderboard=[
            Trial(
                trial_nr=0,
                params={"x": 1},
                score=1.5,
                seconds=0.1,
                failure=None,
            )
        ],
        seconds=0.5,
        stop_reason="search_exhausted",
    )


def make_flat_hpo() -> FlatHyperparameterOptimization:
    return make_hpo_result().to_flat()


def make_flat_evaluation(hpo: FlatHyperparameterOptimization) -> FlatEvaluationData:
    forecasts = pd.DataFrame(
        [
            {
                "location": "A",
                "time_period": "2024-01",
                "horizon_distance": 1,
                "sample": 0,
                "forecast": 10.0,
            },
            {
                "location": "A",
                "time_period": "2024-01",
                "horizon_distance": 1,
                "sample": 1,
                "forecast": 12.0,
            },
        ]
    )
    observations = pd.DataFrame([{"location": "A", "time_period": "2024-01", "disease_cases": 11.0}])
    return FlatEvaluationData(
        forecasts=FlatForecasts.validate(forecasts),
        observations=FlatObserved.validate(observations),
        hpo=hpo,
    )


def test_evaluation_to_flat_converts_runtime_hpo_to_flat_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Evaluation keeps runtime HPO internally and converts it only at the flat serialization boundary."""
    hpo = make_hpo_result()
    backtest = MagicMock()
    backtest.forecasts = []
    backtest.dataset.observations = []
    evaluation = Evaluation(backtest, hpo=hpo)

    forecasts = pd.DataFrame(
        [
            {
                "location": "A",
                "time_period": "2024-01",
                "horizon_distance": 1,
                "sample": 0,
                "forecast": 10.0,
            }
        ]
    )
    observations = pd.DataFrame([{"location": "A", "time_period": "2024-01", "disease_cases": 11.0}])
    monkeypatch.setattr(
        evaluation_module,
        "convert_backtest_to_flat_forecasts",
        lambda _: forecasts,
    )
    monkeypatch.setattr(
        evaluation_module,
        "convert_backtest_observations_to_flat_observations",
        lambda _: observations,
    )

    flat = evaluation.to_flat()

    assert evaluation.get_hpo() is hpo
    assert flat.hpo == hpo.to_flat()


def test_hpo_metadata_is_written_to_evaluation_netcdf(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Flat HPO summary metadata is JSON-encoded into the NetCDF evaluation attribute."""
    hpo = make_flat_hpo()
    flat_data = make_flat_evaluation(hpo)
    backtest = MagicMock()
    backtest.org_units = ["A"]
    backtest.split_periods = ["2024-01"]
    backtest.future_weather_provider = DEFAULT_WEATHER_PROVIDER_ID
    evaluation = Evaluation(backtest)
    monkeypatch.setattr(evaluation, "to_flat", lambda: flat_data)

    monkeypatch.setattr(evaluation_module, "CHAP_VERSION", "2.0.0")
    output = tmp_path / "evaluation-with-hpo.nc"

    evaluation.to_file(
        output,
        model_name="demo-template",
        model_configuration={"user_option_values": {"x": 1}},
        model_version="1.0",
    )

    with xr.open_dataset(output) as dataset:
        stored_hpo = json.loads(dataset.attrs["hpo"])

    assert stored_hpo["searcher"] == "GridSearcher"
    assert stored_hpo["model_template_name"] == "demo-template"
    assert stored_hpo["best_params"] == {"x": 1}
    assert stored_hpo["best_score"] == 1.5
    assert stored_hpo["n_trials"] == 1
    assert stored_hpo["n_successful_trials"] == 1
    assert stored_hpo["n_failed_trials"] == 0


def test_evaluation_from_file_currently_drops_runtime_hpo(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """NetCDF loading documents the current limitation: flat HPO metadata is not rebuilt into runtime HPO."""
    hpo = make_flat_hpo()
    flat_data = make_flat_evaluation(hpo)
    backtest = MagicMock()
    backtest.org_units = ["A"]
    backtest.split_periods = ["2024-01"]
    backtest.future_weather_provider = DEFAULT_WEATHER_PROVIDER_ID
    evaluation = Evaluation(backtest)
    monkeypatch.setattr(evaluation, "to_flat", lambda: flat_data)
    monkeypatch.setattr(evaluation_module, "CHAP_VERSION", "2.0.0")
    output = tmp_path / "evaluation-with-hpo.nc"
    evaluation.to_file(output, model_name="demo-template", model_version="1.0")

    with caplog.at_level(logging.WARNING, logger=evaluation_module.__name__):
        loaded = Evaluation.from_file(output)

    assert loaded.get_hpo() is None
    assert "doesn't yet support converting flat representation back" in caplog.text


def test_evaluation_create_uses_hpo_best_configuration_and_attaches_runtime_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Evaluation.create tunes first, evaluates the winning configuration, and keeps the runtime HPO result."""
    import chap_core.assessment.dataset_splitting as splitting_module
    import chap_core.assessment.prediction_evaluator as prediction_module

    template = MagicMock(name="template")
    template.model_template_config = SimpleNamespace(name="demo-template", version="1.0")
    tuned_estimator = object()
    model_factory = MagicMock(return_value=tuned_estimator)
    template.get_model.return_value = model_factory

    backtest_params = BacktestParams(n_periods=1, n_splits=1, stride=1)
    objective = SimpleNamespace(
        model_template=template,
        direction=SimpleNamespace(value="minimize"),
        metric="rmse",
        backtest_params=backtest_params,
    )
    best_configuration = ModelConfiguration(user_option_values={"x": 1})
    hpo_result = HyperparameterOptimization(
        searcher="GridSearcher",
        model_template_name="demo-template",
        model_template_version="1.0",
        backtest_params=backtest_params,
        metric="rmse",
        search_space={"x": [1, 2]},
        max_trials=None,
        seed=17,
        model_configuration=best_configuration,
        best_params={"x": 1},
        best_score=1.5,
        leaderboard=[
            Trial(
                trial_nr=0,
                params={"x": 1},
                score=1.5,
                seconds=0.1,
                failure=None,
            )
        ],
        seconds=0.5,
        stop_reason="search_exhausted",
    )
    optimizer = HyperparameterOptimizer(
        objective=objective,  # type: ignore[arg-type]
        searcher=GridSearcher(),
        model_configuration=None,
        search_space={"x": [1, 2]},
        max_trials=None,
        seed=17,
    )
    meta_learn_mock = MagicMock(return_value=hpo_result)
    monkeypatch.setattr(optimizer, "meta_learn", meta_learn_mock)

    last_train_period = SimpleNamespace(id="2023-12")
    train_set = SimpleNamespace(period_range=[last_train_period])
    test_generator = object()
    split_mock = MagicMock(return_value=(train_set, test_generator))
    backtest_mock = MagicMock(return_value=[object()])
    monkeypatch.setattr(splitting_module, "train_test_generator", split_mock)
    monkeypatch.setattr(prediction_module, "backtest", backtest_mock)

    monkeypatch.setattr(Evaluation, "calculate_periods_from_years", MagicMock(return_value=0))
    monkeypatch.setattr(Evaluation, "extract_historical_observations", MagicMock(return_value=[]))
    final_evaluation = object()
    from_samples_mock = MagicMock(return_value=final_evaluation)
    monkeypatch.setattr(Evaluation, "from_samples_with_truth", from_samples_mock)

    configured_model = SimpleNamespace(id="configured-model")
    dataset = object()

    result = Evaluation.create(
        configured_model=configured_model,  # type: ignore[arg-type]
        estimator=optimizer,
        dataset=dataset,  # type: ignore[arg-type]
        backtest_params=backtest_params,
        historical_context_years=2,
    )

    assert result is final_evaluation
    meta_learn_mock.assert_called_once_with(train_set)
    template.get_model.assert_called_once_with(best_configuration, prediction_length=backtest_params.n_periods)
    model_factory.assert_called_once_with()

    backtest_kwargs = backtest_mock.call_args.kwargs
    assert backtest_kwargs["estimator"] is tuned_estimator
    assert backtest_kwargs["train_set"] is train_set
    assert backtest_kwargs["test_generator"] is test_generator

    from_samples_kwargs = from_samples_mock.call_args.kwargs
    assert from_samples_kwargs["hpo"] is hpo_result
