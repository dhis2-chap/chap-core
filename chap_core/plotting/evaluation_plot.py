import altair as alt
from sqlmodel import Field

from chap_core.assessment.evaluation import Evaluation
from chap_core.assessment.metric_plots import MetricPlotBase
from chap_core.assessment.metrics.base import Metric
from chap_core.database.base_tables import DBModel
from chap_core.database.tables import Backtest


class VisualizationInfo(DBModel):
    """Catalogue entry for one available metric visualisation (returned by `/v1/visualization/metrics/{backtest_id}` and similar)."""

    id: str = Field(description="Canonical plot identifier used in URLs.")
    display_name: str = Field(description="Human-friendly plot name shown in pickers.")
    description: str = Field(description="Short paragraph explaining what the plot shows.")


def make_plot_from_backtest_object(
    backtest: Backtest, plotting_class: type[MetricPlotBase], metric: Metric, geojson: dict | None = None
) -> dict:
    evaluation = Evaluation.from_backtest(backtest)
    flat_data = evaluation.to_flat()
    metric_data = metric.get_detailed_metric(flat_data.observations, flat_data.forecasts)
    return plotting_class(metric_data, geojson).plot_spec()


def make_plot_from_evaluation_object(
    evaluation: Evaluation, plotting_class: type[MetricPlotBase], metric: Metric, geojson: dict | None = None
) -> alt.Chart:
    flat_data = evaluation.to_flat()
    metric_data = metric.get_detailed_metric(flat_data.observations, flat_data.forecasts)
    return plotting_class(metric_data, geojson).plot()
