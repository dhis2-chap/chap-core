"""Response models of the benchmark leaderboard, `GET /v1/crud/backtest-specifications/{id}`.

They live apart from `data_models` because they describe the metrics, and the metrics
package imports the evaluation code, which imports `data_models`.
"""

from pydantic import Field

from chap_core.api_types import BacktestParams
from chap_core.assessment.metrics import available_metrics, get_comparison_description
from chap_core.assessment.metrics.base import OptimizationDirection, TargetBehavior
from chap_core.database.base_tables import DBModel
from chap_core.database.settings import Settings
from chap_core.database.tables import BacktestRead, ConfiguredModelRead, DataSetMeta, SpecificationBaselineStatus


class MetricInfo(DBModel):
    """Catalogue entry for one scoring metric (CRPS, MAE, ...)."""

    id: str = Field(description="Canonical metric identifier used in URLs and request bodies.")
    display_name: str = Field(description="Human-friendly metric name shown in pickers.")
    description: str = Field(default="", description="Short paragraph explaining what the metric measures.")
    sort_order: int = Field(description="Display and ranking priority, starting at zero; lower comes first.")
    is_selected: bool = Field(description="Whether this metric is selected by the installation's settings.")
    unit: str | None = Field(default=None, description="Display suffix for the raw score, e.g. '%' for MAPE.")
    target: float | None = Field(
        default=None,
        description="Ideal value in raw score units, e.g. 0.8 for 80% coverage. Null when no fixed target applies.",
    )
    target_behavior: TargetBehavior = Field(
        default=TargetBehavior.CLOSEST,
        description=(
            "How to judge a score against ``target``, only meaningful when ``target`` is set. "
            "'closest' means deviating in either direction is worse. 'at_least' means higher is "
            "better up to the target and flat above it, so only scores below the target should be "
            "flagged as bad."
        ),
    )
    optimization_direction: OptimizationDirection | None = Field(
        default=None,
        description=(
            "Whether a lower ('minimize') or higher ('maximize') score is better. "
            "Null for metrics where neither direction is better; those set ``target`` instead, "
            "and ``target_behavior`` says how to judge a score against it."
        ),
    )
    comparison_op: str | None = Field(
        default=None,
        description=(
            "Name of the operator used to compare a score with a baseline or comparison model's score: "
            "'skill_ratio' (relative improvement, 1 - score / reference for error scores), 'difference' "
            "(improvement in the metric's own units) or 'target_distance' (how much closer to ``target``). "
            "Positive always means better than the reference. Null when the metric has no comparison."
        ),
    )
    comparison_description: str | None = Field(
        default=None,
        description="One-line description of what the metric's comparison value means. Null without a comparison.",
    )


def list_metric_info(settings: Settings) -> list[MetricInfo]:
    """Describe every metric in the configured order, retaining unselected metrics."""
    metrics = []
    for rank, metric_id in enumerate(settings.metric_sort_order):
        metric_cls = available_metrics[metric_id]
        metrics.append(
            MetricInfo(
                id=metric_id,
                sort_order=rank,
                is_selected=metric_id in settings.selected_metrics,
                display_name=metric_cls().get_name(),
                description=metric_cls().get_description(),
                optimization_direction=metric_cls.spec.optimization_direction,
                target_behavior=metric_cls.spec.target_behavior,
                unit=metric_cls.spec.unit,
                target=metric_cls.spec.target,
                comparison_op=metric_cls.spec.comparison_op,
                comparison_description=get_comparison_description(metric_cls.spec.comparison_op)
                if metric_cls.spec.comparison_op
                else None,
            )
        )
    return metrics


class ReferenceModelResult(DBModel):
    """A baseline or comparison model's result on a specification, the reference other backtests are compared with."""

    configured_model: ConfiguredModelRead = Field(description="The reference model.")
    status: SpecificationBaselineStatus = Field(
        description="`done` once the model's backtest is stored. A baseline is `pending` until it has run, "
        "`not_applicable` if it rejects the specification's data, and `failed` if its run failed."
    )
    message: str | None = Field(default=None, description="Why the baseline is not applicable or failed.")
    backtest_id: int | None = Field(default=None, description="Id of the model's backtest once it is done.")
    aggregate_metrics: dict[str, float] = Field(
        default_factory=dict, description="Map of metric id to aggregated score; empty until the model is done."
    )


class ReferenceComparison(DBModel):
    """A backtest's aggregated metrics compared with one reference model's."""

    configured_model_id: int = Field(description="Id of the reference model's configured model.")
    values: dict[str, float | None] = Field(
        description="Map of metric id to the metric's comparison value, positive meaning better than the reference. "
        "Only metrics with a comparison operator are included; null when the comparison is undefined."
    )


class LeaderboardBacktestRead(BacktestRead):
    """A backtest row of the leaderboard: `BacktestRead` plus its comparisons with the reference models."""

    comparisons: list[ReferenceComparison] = Field(
        default_factory=list, description="One entry per reference model that is done, in the order they are listed."
    )


class BacktestSpecificationRead(BacktestParams):
    """A specification with every backtest under it: the benchmark leaderboard in one response."""

    target_column: str = Field(default="disease_cases", description="Dataset column to evaluate as the target.")

    id: int = Field(description="Primary key of the specification.")
    dataset: DataSetMeta = Field(description="Slim summary of the dataset the specification evaluates against.")
    org_units: list[str] = Field(description="Org units the evaluation runs over, resolved from the dataset.")
    backtests: list[LeaderboardBacktestRead] = Field(
        description="Every backtest that ran under this specification, newest first. Comparable by construction."
    )
    baselines: list[ReferenceModelResult] = Field(
        default_factory=list, description="The current version of each built-in baseline model on this specification."
    )
    comparison_models: list[ReferenceModelResult] | None = Field(
        default=None,
        description="The newest backtest of each comparison model on this specification; "
        "null unless `includeComparisonModels` is set.",
    )
    metrics: list[MetricInfo] = Field(
        default_factory=list,
        description="Every metric in configured ranking order, marked with isSelected for the default display. "
        "Sort by the selected metrics in this order, using each metric's direction or target.",
    )
    default_baseline_model: str = Field(description="Built-in baseline name to use for ratios and model promotion.")
