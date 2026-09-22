from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from chap_core.api_types import BacktestParams
from chap_core.database.model_templates_and_config_tables import ModelConfiguration
from chap_core.hpo.objective import Objective


class FakeModelTemplateDB:
    def __init__(self, **kwargs: object) -> None:
        self.__dict__.update(kwargs)


class FakeConfiguredModelDB:
    def __init__(self, **kwargs: object) -> None:
        self.__dict__.update(kwargs)


def make_template() -> tuple[MagicMock, MagicMock, object]:
    template = MagicMock(name="model_template")
    template.model_template_config = SimpleNamespace(name="demo-template", version="1.2.3")
    estimator = object()
    model_factory = MagicMock(name="configured_model_factory", return_value=estimator)
    template.get_model.return_value = model_factory
    return template, model_factory, estimator


def patch_runtime_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    *,
    metric_value: float | None = 2.5,
    evaluation: MagicMock | None = None,
) -> tuple[MagicMock, MagicMock, MagicMock]:
    import chap_core.assessment.evaluation as evaluation_module
    import chap_core.assessment.metrics as metrics_module
    import chap_core.database.model_templates_and_config_tables as db_module

    evaluation = evaluation or MagicMock(name="evaluation")
    create_mock = MagicMock(return_value=evaluation)
    calculate_metrics_mock = MagicMock(return_value={"rmse": metric_value})

    monkeypatch.setattr(db_module, "ModelTemplateDB", FakeModelTemplateDB)
    monkeypatch.setattr(db_module, "ConfiguredModelDB", FakeConfiguredModelDB)
    monkeypatch.setattr(evaluation_module.Evaluation, "create", create_mock)
    monkeypatch.setattr(metrics_module, "calculate_metrics", calculate_metrics_mock)

    return evaluation, create_mock, calculate_metrics_mock


def test_objective_defaults_to_rmse_and_minimize_direction() -> None:
    """The default objective metric is RMSE and uses CHAP's registered optimization direction."""
    template, _, _ = make_template()

    objective = Objective(
        model_template=template,
        backtest_params=BacktestParams(n_periods=2, n_splits=3, stride=1),
    )

    assert objective.metric == "rmse"
    assert objective.direction.value == "minimize"


def test_objective_runs_configured_estimator_evaluation_and_selected_metric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Objective wires the complete model configuration into CHAP evaluation and returns the chosen scalar metric."""
    template, model_factory, estimator = make_template()
    evaluation, create_mock, calculate_metrics_mock = patch_runtime_dependencies(monkeypatch)
    monkeypatch.setattr("chap_core.hpo.objective.generate_short_id", lambda: "abc123")

    backtest_params = BacktestParams(n_periods=2, n_splits=3, stride=1)
    configuration = ModelConfiguration(
        user_option_values={"alpha": 0.25, "fixed": 10},
        additional_continuous_covariates=["rainfall"],
    )
    dataset = object()
    objective = Objective(
        model_template=template,
        backtest_params=backtest_params,
        metric="rmse",
        historical_context_years=4,
    )

    score = objective(configuration, dataset)  # type: ignore[arg-type]

    assert score == pytest.approx(2.5)
    template.get_model.assert_called_once_with(configuration, prediction_length=backtest_params.n_periods)
    model_factory.assert_called_once_with()

    create_kwargs = create_mock.call_args.kwargs
    assert create_kwargs["estimator"] is estimator
    assert create_kwargs["dataset"] is dataset
    assert create_kwargs["backtest_params"] == backtest_params
    assert create_kwargs["backtest_name"] == "hpo_validation_abc123"
    assert create_kwargs["historical_context_years"] == 4

    configured_model = create_kwargs["configured_model"]
    assert configured_model.id == "hpo_abc123"
    assert configured_model.model_template_id == "demo-template"
    assert configured_model.user_option_values == {"alpha": 0.25, "fixed": 10}
    assert configured_model.additional_continuous_covariates == ["rainfall"]

    calculate_metrics_mock.assert_called_once_with(
        evaluation=evaluation,
        metric_ids=["rmse"],
    )


def test_objective_exports_validation_when_eval_output_dir_is_configured(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Optional objective diagnostics write the validation evaluation with configuration/version metadata."""
    template, _, _ = make_template()
    evaluation, _, _ = patch_runtime_dependencies(monkeypatch)
    monkeypatch.setattr("chap_core.hpo.objective.generate_short_id", lambda: "run42")

    def write_file(*, filepath: Path, **_: object) -> None:
        Path(filepath).write_text("written", encoding="utf-8")

    evaluation.to_file.side_effect = write_file
    output_dir = tmp_path / "nested" / "validations"
    configuration = ModelConfiguration(user_option_values={"x": 7})
    objective = Objective(
        model_template=template,
        backtest_params=BacktestParams(n_periods=1, n_splits=1, stride=1),
        eval_output_dir=output_dir,
    )

    objective(configuration, object())  # type: ignore[arg-type]

    expected = output_dir / "hpo_validation_run42.nc"
    assert expected.read_text(encoding="utf-8") == "written"
    evaluation.to_file.assert_called_once_with(
        filepath=expected,
        model_name="hpo_config_run42",
        model_configuration=configuration.model_dump(),
        model_version="1.2.3",
    )


def test_objective_raises_when_metric_cannot_be_calculated(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing metric value fails the trial instead of being silently converted to an invalid score."""
    template, _, _ = make_template()
    patch_runtime_dependencies(monkeypatch, metric_value=None)
    configuration = ModelConfiguration(user_option_values={"x": 1})
    objective = Objective(
        model_template=template,
        backtest_params=BacktestParams(n_periods=1, n_splits=1, stride=1),
    )

    with pytest.raises(ValueError, match="Metric rmse could not be calculated"):
        objective(configuration, object())  # type: ignore[arg-type]


def test_objective_propagates_evaluation_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Model/backtest failures propagate to HyperparameterOptimizer, which owns trial-failure handling."""
    template, _, _ = make_template()
    _, create_mock, calculate_metrics_mock = patch_runtime_dependencies(monkeypatch)
    create_mock.side_effect = RuntimeError("backtest failed")
    configuration = ModelConfiguration(user_option_values={"x": 1})
    objective = Objective(
        model_template=template,
        backtest_params=BacktestParams(n_periods=1, n_splits=1, stride=1),
    )

    with pytest.raises(RuntimeError, match="backtest failed"):
        objective(configuration, object())  # type: ignore[arg-type]

    calculate_metrics_mock.assert_not_called()


@pytest.mark.parametrize("metric", ["coverage_10_90", "sensitivity"])
def test_objective_rejects_metric_not_supported_as_direct_hpo_objective(metric: str) -> None:
    """Metrics that are valid for reporting but not direct HPO objectives fail before a backtest starts."""
    template, _, _ = make_template()

    with pytest.raises(ValueError, match="not defined as a direct HPO objective"):
        Objective(
            model_template=template,
            backtest_params=BacktestParams(n_periods=1, n_splits=1, stride=1),
            metric=metric,
        )
