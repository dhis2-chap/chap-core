from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from cyclopts import App

from chap_core.api_types import BacktestParams, EstimatorMode, EstimatorOptions, SearcherType
from chap_core.cli_endpoints._common import get_hpo_estimator
from chap_core.cli_endpoints.evaluate import register_commands
from chap_core.hpo.hyperparameter_optimizer import HyperparameterOptimizer
from chap_core.hpo.search_space import DEFAULT_HPO_TRIALS, Int, Float
from chap_core.hpo.searcher import GridSearcher, RandomSearcher, TPESearcher


class FakeDBRecord:
    def __init__(self, **kwargs: object) -> None:
        self.__dict__.update(kwargs)


def make_template(search_space: dict | None) -> MagicMock:
    template = MagicMock(name="model_template")
    template.model_template_config = SimpleNamespace(
        name="demo-template",
        version="1.0",
        hpo_search_space=search_space,
        min_prediction_periods=None,
        max_prediction_periods=None,
    )
    template.__enter__.return_value = template
    template.__exit__.return_value = False
    return template


def test_get_hpo_estimator_defaults_to_tpe_and_default_budget() -> None:
    """Factory defaults match the user-facing HPO defaults when only a template search space is supplied."""
    template = make_template({"x": {"low": 1, "high": 3, "type": "int"}})

    optimizer = get_hpo_estimator(
        template=template,
        configuration=None,
        backtest_params=BacktestParams(n_periods=1, n_splits=1, stride=1),
        options=EstimatorOptions(mode=EstimatorMode.HPO),
    )

    assert isinstance(optimizer, HyperparameterOptimizer)
    assert isinstance(optimizer._searcher, TPESearcher)
    assert optimizer._search_space == {"x": Int(1, 3)}
    assert optimizer._max_trials == DEFAULT_HPO_TRIALS


def test_get_hpo_estimator_grid_is_exhaustive_by_default() -> None:
    """Selecting grid search leaves max_trials unset so the whole finite grid can be evaluated."""
    template = make_template({"x": {"values": [1, 2, 3]}})

    optimizer = get_hpo_estimator(
        template=template,
        configuration=None,
        backtest_params=BacktestParams(n_periods=1, n_splits=1, stride=1),
        options=EstimatorOptions(mode=EstimatorMode.HPO, searcher=SearcherType.GRID),
    )

    assert isinstance(optimizer._searcher, GridSearcher)
    assert optimizer._max_trials is None


def test_get_hpo_estimator_explicit_yaml_overrides_template_space(tmp_path: Path) -> None:
    """--estimator-options.search-space takes precedence over the model template's built-in HPO space."""
    template = make_template({"template_only": {"values": [1]}})
    search_space_file = tmp_path / "search-space.yaml"
    search_space_file.write_text(
        "x:\n  values: [10, 20]\n",
        encoding="utf-8",
    )

    optimizer = get_hpo_estimator(
        template=template,
        configuration=None,
        backtest_params=BacktestParams(n_periods=1, n_splits=1, stride=1),
        options=EstimatorOptions(
            mode=EstimatorMode.HPO,
            search_space=search_space_file,
            searcher=SearcherType.RANDOM,
            max_trials=7,
            seed=19,
        ),
    )

    assert isinstance(optimizer._searcher, RandomSearcher)
    assert optimizer._search_space == {"x": [10, 20]}
    assert optimizer._max_trials == 7
    assert optimizer._seed == 19


def test_get_hpo_estimator_rejects_missing_search_space() -> None:
    """HPO cannot start when neither an explicit YAML file nor template metadata defines a search space."""
    template = make_template(None)

    with pytest.raises(ValueError, match="HPO search space YAML must define a non-empty mapping"):
        get_hpo_estimator(
            template=template,
            configuration=None,
            backtest_params=BacktestParams(n_periods=1, n_splits=1, stride=1),
            options=EstimatorOptions(mode=EstimatorMode.HPO),
        )


def test_eval_cli_parses_hpo_options_and_passes_optimizer_to_evaluation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The public `chap eval` option syntax constructs the requested HPO estimator before evaluation."""
    import chap_core.assessment.eval_tracking as tracking_module
    import chap_core.assessment.evaluation as evaluation_module
    import chap_core.cli_endpoints.evaluate as evaluate_module
    import chap_core.database.model_templates_and_config_tables as db_module
    import chap_core.log_config as log_config_module
    import chap_core.models.model_template as model_template_module
    import chap_core.rest_api.db_worker_functions as db_worker_module

    search_space_file = tmp_path / "search-space.yaml"
    search_space_file.write_text("x:\n  values: [1, 2]\n", encoding="utf-8")
    output_file = tmp_path / "hpo-evaluation.nc"

    template = make_template(None)
    monkeypatch.setattr(
        model_template_module.ModelTemplate,
        "from_directory_or_github_url",
        MagicMock(return_value=template),
    )
    monkeypatch.setattr(evaluate_module, "resolve_csv_path", lambda _: (Path("input.csv"), None))
    monkeypatch.setattr(evaluate_module, "discover_geojson", lambda _: None)
    dataset = MagicMock(name="dataset")
    monkeypatch.setattr(evaluate_module, "load_dataset_from_csv", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(
        db_worker_module,
        "validate_and_filter_dataset_for_evaluation",
        lambda value, **kwargs: value,
    )
    monkeypatch.setattr(evaluate_module, "get_configuration", lambda _: None)
    monkeypatch.setattr(evaluate_module, "warn_unused_covariates", lambda *args, **kwargs: None)
    monkeypatch.setattr(log_config_module, "initialize_logging", lambda *args, **kwargs: None)

    monkeypatch.setattr(db_module, "ModelTemplateDB", FakeDBRecord)
    monkeypatch.setattr(db_module, "ConfiguredModelDB", FakeDBRecord)

    fake_evaluation = MagicMock(name="evaluation")

    def write_fake_output(*, filepath: Path, **_: object) -> None:
        filepath.touch()

    fake_evaluation.to_file.side_effect = write_fake_output
    create_mock = MagicMock(return_value=fake_evaluation)
    monkeypatch.setattr(evaluation_module.Evaluation, "create", create_mock)

    tracker = MagicMock(name="tracker")

    @contextmanager
    def fake_tracked_eval_run(**_: object):
        yield tracker

    monkeypatch.setattr(tracking_module, "load_model_configuration", lambda _: None)
    monkeypatch.setattr(tracking_module, "tracked_eval_run", fake_tracked_eval_run)

    app = App()
    register_commands(app)
    result = app(
        [
            "eval",
            "--model-name",
            "dummy-model",
            "--dataset-csv",
            "input.csv",
            "--output-file",
            str(output_file),
            "--backtest-params.n-periods",
            "1",
            "--backtest-params.n-splits",
            "1",
            "--estimator-options.mode",
            "hpo",
            "--estimator-options.search-space",
            str(search_space_file),
            "--estimator-options.searcher",
            "random",
            "--estimator-options.max-trials",
            "2",
            "--estimator-options.seed",
            "17",
        ],
        exit_on_error=False,
        result_action="return_value",
    )

    assert result is None
    estimator = create_mock.call_args.kwargs["estimator"]
    assert isinstance(estimator, HyperparameterOptimizer)
    assert isinstance(estimator._searcher, RandomSearcher)
    assert estimator._search_space == {"x": [1, 2]}
    assert estimator._max_trials == 2
    assert estimator._seed == 17
    assert create_mock.call_args.kwargs["backtest_params"].n_periods == 1
    assert create_mock.call_args.kwargs["backtest_params"].n_splits == 1
    assert output_file.exists()


def test_get_hpo_estimator_uses_template_hpo_search_space_when_no_yaml_is_given() -> None:
    """Without an explicit search-space YAML, HPO uses the search space from the model template."""
    template = make_template(
        {
            "learning_rate": {
                "type": "float",
                "low": 0.001,
                "high": 0.01,
                "log": True,
            },
            "max_epochs": {
                "type": "int",
                "low": 1,
                "high": 3,
            },
            "batch_size": {
                "values": [32, 64],
            },
        }
    )

    optimizer = get_hpo_estimator(
        template=template,
        configuration=None,
        backtest_params=BacktestParams(
            n_periods=1,
            n_splits=1,
            stride=1,
        ),
        options=EstimatorOptions(
            mode=EstimatorMode.HPO,
            # Deliberately no search_space argument.
        ),
    )

    assert optimizer._search_space == {
        "learning_rate": Float(
            low=0.001,
            high=0.01,
            step=None,
            log=True,
        ),
        "max_epochs": Int(
            low=1,
            high=3,
            step=1,
            log=False,
        ),
        "batch_size": [32, 64],
    }
