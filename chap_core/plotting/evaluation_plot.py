import altair as alt

from chap_core.assessment.evaluation import Evaluation
from chap_core.assessment.metric_plots import MetricPlotBase
from chap_core.assessment.metrics.base import Metric
from chap_core.database.base_tables import DBModel
from chap_core.database.tables import BackTest


class VisualizationInfo(DBModel):
    id: str
    display_name: str
    description: str


def make_plot_from_backtest_object(
    backtest: BackTest, plotting_class: type[MetricPlotBase], metric: Metric, geojson: dict | None = None
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
