from contextlib import ExitStack
from unittest.mock import MagicMock, patch

from chap_core.api import forecast
import pytest
from chap_core.util import docker_available
from chap_core.cli_endpoints.evaluate import eval_cmd
from chap_core.cli_endpoints.utils import sanity_check_model


@pytest.mark.skipif(not docker_available(), reason="Docker not available")
@pytest.mark.skip(reason="Failing on CI")
def test_forecast_github_model():
    repo_url = "https://github.com/knutdrand/external_rmodel_example.git"
    results = forecast("external", "hydromet_5_filtered", 12, repo_url)


def test_eval_cmd(tmp_path):
    from chap_core.file_io.example_data_set import datasets
    from chap_core.api_types import BacktestParams, RunConfig

    # Export hydromet dataset to CSV for testing
    dataset = datasets["hydromet_5_filtered"].load()
    csv_path = tmp_path / "test_data.csv"
    dataset.to_csv(csv_path)

    # Prepare parameters
    backtest_params = BacktestParams(n_periods=3, n_splits=2, stride=1)
    run_config = RunConfig()

    # Run eval_cmd with CSV
    output_file = tmp_path / "evaluation.nc"
    eval_cmd(
        model_name="https://github.com/dhis2-chap/minimalist_example_lag",
        dataset_csv=csv_path,
        output_file=output_file,
        backtest_params=backtest_params,
        run_config=run_config,
    )

    # Verify output file was created
    assert output_file.exists()


def _make_fake_estimator(min_prediction_length, max_prediction_length):
    from chap_core.database.model_templates_and_config_tables import ModelTemplateInformation

    info = ModelTemplateInformation(
        min_prediction_length=min_prediction_length,
        max_prediction_length=max_prediction_length,
    )
    estimator = MagicMock(name="FakeEstimator")
    estimator.model_information = info
    return estimator


def _patched_eval_chain(fake_estimator):
    """Stack mocks for the parts of eval_cmd that aren't under test, returning
    the Evaluation mock so the caller can inspect ``Evaluation.create`` calls."""
    template_cm = MagicMock(name="ModelTemplate")
    template_cm.__enter__.return_value = template_cm
    template_cm.__exit__.return_value = False
    template_cm.model_template_config.name = "test_model"
    template_cm.model_template_config.version = "1"

    stack = ExitStack()
    stack.enter_context(
        patch(
            "chap_core.cli_endpoints.evaluate.resolve_csv_path",
            return_value=("dummy.csv", None),
        )
    )
    stack.enter_context(patch("chap_core.cli_endpoints.evaluate.discover_geojson", return_value=None))
    stack.enter_context(
        patch(
            "chap_core.cli_endpoints.evaluate.load_dataset_from_csv",
            return_value=MagicMock(name="DataSet"),
        )
    )
    stack.enter_context(patch("chap_core.log_config.initialize_logging"))
    mt_mock = stack.enter_context(patch("chap_core.models.model_template.ModelTemplate"))
    mt_mock.from_directory_or_github_url.return_value = template_cm
    stack.enter_context(
        patch(
            "chap_core.cli_endpoints.evaluate.get_estimator",
            return_value=(fake_estimator, None),
        )
    )
    eval_mock = stack.enter_context(patch("chap_core.assessment.evaluation.Evaluation"))
    eval_mock.create.return_value = MagicMock(name="EvaluationInstance")
    return stack, eval_mock


def test_eval_cmd_raises_when_n_periods_below_min_prediction_length(tmp_path):
    """Dispatch in evaluate.py guards against horizons shorter than the model's declared min."""
    from chap_core.api_types import BacktestParams, RunConfig

    fake_estimator = _make_fake_estimator(min_prediction_length=5, max_prediction_length=10)
    stack, _ = _patched_eval_chain(fake_estimator)
    with stack:
        with pytest.raises(ValueError, match="minimum prediction length"):
            eval_cmd(
                model_name="dummy",
                dataset_csv="dummy.csv",
                output_file=tmp_path / "out.nc",
                backtest_params=BacktestParams(n_periods=3, n_splits=2, stride=1),
                run_config=RunConfig(),
            )


def test_eval_cmd_wraps_in_extended_predictor_when_n_periods_above_max(tmp_path):
    """When the requested horizon exceeds the model's declared max, the dispatch wraps
    the estimator in ExtendedPredictor and forwards the wrapped estimator to evaluation."""
    from chap_core.api_types import BacktestParams, RunConfig
    from chap_core.external.ExtendedPredictor import ExtendedPredictor

    fake_estimator = _make_fake_estimator(min_prediction_length=1, max_prediction_length=2)
    stack, eval_mock = _patched_eval_chain(fake_estimator)
    with stack:
        eval_cmd(
            model_name="dummy",
            dataset_csv="dummy.csv",
            output_file=tmp_path / "out.nc",
            backtest_params=BacktestParams(n_periods=5, n_splits=2, stride=1),
            run_config=RunConfig(),
        )

    assert eval_mock.create.call_count == 1
    forwarded = eval_mock.create.call_args.kwargs["estimator"]
    assert isinstance(forwarded, ExtendedPredictor)


def test_eval_cmd_does_not_wrap_when_bounds_unspecified(tmp_path):
    """When the model declares neither min nor max, dispatch logs a warning and forwards
    the original estimator unchanged."""
    from chap_core.api_types import BacktestParams, RunConfig

    fake_estimator = _make_fake_estimator(min_prediction_length=None, max_prediction_length=None)
    stack, eval_mock = _patched_eval_chain(fake_estimator)
    with stack:
        eval_cmd(
            model_name="dummy",
            dataset_csv="dummy.csv",
            output_file=tmp_path / "out.nc",
            backtest_params=BacktestParams(n_periods=3, n_splits=2, stride=1),
            run_config=RunConfig(),
        )

    assert eval_mock.create.call_count == 1
    forwarded = eval_mock.create.call_args.kwargs["estimator"]
    assert forwarded is fake_estimator


def test_eval_cmd_with_data_source_mapping(tmp_path):
    import json

    import pandas as pd

    from chap_core.api_types import BacktestParams, RunConfig
    from chap_core.file_io.example_data_set import datasets

    # Export hydromet dataset to CSV for testing
    dataset = datasets["hydromet_5_filtered"].load()
    csv_path = tmp_path / "test_data.csv"
    dataset.to_csv(csv_path)

    # Rename columns to simulate a CSV with different column names
    df = pd.read_csv(csv_path)
    df.rename(columns={"rainfall": "precip", "disease_cases": "cases"}, inplace=True)
    renamed_csv_path = tmp_path / "renamed_data.csv"
    df.to_csv(renamed_csv_path, index=False)

    # Create mapping JSON file
    mapping = {"rainfall": "precip", "disease_cases": "cases"}
    mapping_path = tmp_path / "column_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(mapping, f)

    # Prepare parameters
    backtest_params = BacktestParams(n_periods=3, n_splits=2, stride=1)
    run_config = RunConfig()

    # Run eval_cmd with the mapping
    output_file = tmp_path / "evaluation.nc"
    eval_cmd(
        model_name="https://github.com/dhis2-chap/minimalist_example_lag",
        dataset_csv=renamed_csv_path,
        output_file=output_file,
        backtest_params=backtest_params,
        run_config=run_config,
        data_source_mapping=mapping_path,
    )

    # Verify output file was created
    assert output_file.exists()
