import abc
from typing import Optional

import altair as alt
from chap_core.assessment.evaluation import Evaluation
from chap_core.assessment.flat_representations import FlatMetric
from chap_core.assessment.metrics.base import MetricBase
from chap_core.database.base_tables import DBModel
from chap_core.database.tables import BackTest

alt.renderers.enable("browser")


class MetricPlotV2(abc.ABC):
    """
    Represents types of metrics plots, that always start from raw FlatMetric data.
    Differnet plots can process this data in the way they want to produce a plot
    """

    def __init__(self, metric_data: FlatMetric, geojson: Optional[dict] = None):
        self._metric_data = metric_data

    def plot(self, title="Mean metric by horizon") -> alt.Chart:
        return self.plot_from_df(title=title)

    @abc.abstractmethod
    def plot_from_df(self, title: str = "") -> alt.Chart:
        pass

    def plot_spec(self) -> dict:
        chart = self.plot()
        return chart.to_dict(format="vega")


class VisualizationInfo(DBModel):
    id: str
    display_name: str
    description: str


class MetricByHorizonAndLocationMean(MetricPlotV2):
    visualization_info = VisualizationInfo(
        id="metric_by_horizon",
        display_name="Horizon Plot",
        description="Shows the aggregated metric by forecast horizon",
    )

    def plot_from_df(self):
        df = self._metric_data
        adf = df.groupby(["horizon_distance", "location"]).agg({"metric": "mean"}).reset_index()
        print(adf)
        chart = (
            alt.Chart(adf)
            .mark_bar(point=True)
            .encode(
                x=alt.X("horizon_distance:O", title="Horizon (periods ahead)"),
                y=alt.Y("metric:Q", title="Mean Metric Value"),
                tooltip=["horizon_distance", "location", "metric"],
            )
            .properties(width=300, height=230, title="Mean Metric by Horizon")
        )

        return chart


class MetricByHorizonV2Mean(MetricPlotV2):
    visualization_info = VisualizationInfo(
        id="metric_by_horizon",
        display_name="Horizon Plot",
        description="Shows the aggregated metric by forecast horizon",
    )

    def plot_from_df(self, title="Mean metric by horizon"):
        df = self._metric_data
        adf = df.groupby(["horizon_distance"]).agg({"metric": "mean"}).reset_index()
        chart = (
            alt.Chart(adf)
            .mark_bar(point=True)
            .encode(
                x=alt.X("horizon_distance:O", title="Horizon (periods ahead)"),
                y=alt.Y("metric:Q", title="Mean Metric Value"),
                tooltip=["horizon_distance", "metric"],
            )
            .properties(width=300, height=230, title=title)
        )

        return chart


class MetricByHorizonV2Sum(MetricPlotV2):
    visualization_info = VisualizationInfo(
        id="metric_by_horizon_sum",
        display_name="Horizon Plot (sum)",
        description="Sums metric across locations per forecast horizon",
    )

    def plot_from_df(self):
        df = self._metric_data
        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X("horizon_distance:O", title="Horizon (periods ahead)"),
                y=alt.Y("sum(metric):Q", title="Samples above truth (count)"),
                tooltip=[
                    alt.Tooltip("horizon_distance:O", title="Horizon"),
                    alt.Tooltip("sum(metric):Q", title="Count"),
                ],
            )
            .properties(width=300, height=230, title="Samples above truth by horizon")
        )

        return chart


class MetricByTimePeriodAndLocationV2Mean(MetricPlotV2):
    visualization_info = VisualizationInfo(
        id="metric_by_time_period",
        display_name="Time Period Plot",
        description="Shows the aggregated metric by time period (per location)",
    )

    def plot_from_df(self, title="Mean metric by location and time period") -> alt.Chart:
        df = self._metric_data
        adf = df.groupby(["time_period", "location"]).agg({"metric": "mean"}).reset_index()
        chart = (
            alt.Chart(adf)
            .mark_line(point=True)
            .encode(
                x=alt.X("time_period:O", title="Time period"),
                y=alt.Y("metric:Q", title="Mean Metric Value"),
                color=alt.Color("location:N", title="Location"),
                tooltip=["time_period", "location", "metric"],
            )
            .properties(width=300, height=230, title=title)
        )

        return chart


class MetricByTimePeriodV2Sum(MetricPlotV2):
    visualization_info = VisualizationInfo(
        id="metric_by_time_sum",
        display_name="Horizon Plot (sum)",
        description="Sums metric across locations per forecast horizon",
    )

    def plot_from_df(self):
        df = self._metric_data
        chart = (
            alt.Chart(df)
            .mark_line()
            .encode(
                x=alt.X("time_period:O", title="Time Period"),
                y=alt.Y("sum(metric):Q", title="Samples above truth (count)"),
                color=alt.Color("location:N", title="Location"),
                tooltip=[
                    alt.Tooltip("time_period:O", title="Time Period"),
                    alt.Tooltip("sum(metric):Q", title="Count"),
                ],
            )
            .properties(width=300, height=230, title="Samples above truth by time period")
        )

        return chart


class MetricByTimePeriodV2Mean(MetricPlotV2):
    visualization_info = VisualizationInfo(
        id="metric_by_time_mean",
        display_name="Metric by time (mean)",
        description="Mean metric across locations and horizons per time period",
    )

    def plot_from_df(self, title="Mean metric by time period"):
        df = self._metric_data
        df = df.groupby(["time_period"]).agg({"metric": "mean"}).reset_index()
        chart = (
            alt.Chart(df)
            .mark_line()
            .encode(
                x=alt.X("time_period:O", title="Time Period"),
                y=alt.Y("mean(metric):Q", title="Mean Metric Value"),
                tooltip=[
                    alt.Tooltip("time_period:O", title="Time Period"),
                    alt.Tooltip("mean(metric):Q", title="Count"),
                ],
            )
            .properties(width=300, height=230, title=title)
        )

        return chart


class MetricMapV2(MetricPlotV2):
    visualization_info = VisualizationInfo(
        id="metric_map", display_name="Map", description="Shows a map of aggregated metrics per org unit"
    )

    def __init__(self, metric_data: FlatMetric, geojson: Optional[dict] = None):
        super().__init__(metric_data, geojson)
        self._geojson = geojson

    def plot_from_df(self, title="Metric Map by location") -> alt.Chart:
        # Get the metric data DataFrame
        df = self._metric_data

        # Aggregate metrics by location (average across all time periods and horizons)
        agg_df = df.groupby("location").agg({"metric": "mean"}).reset_index()
        agg_df.rename(columns={"location": "org_unit", "metric": "value"}, inplace=True)

        # Create map visualization with geojson
        geojson_data = self._geojson

        # Build Altair map chart
        chart = (
            alt.Chart(alt.Data(values=geojson_data["features"]))
            .mark_geoshape(stroke="black", strokeWidth=0.5)
            .encode(
                color=alt.Color("value:Q", scale=alt.Scale(scheme="reds"), title="Metric Value"),
                tooltip=[alt.Tooltip("properties.name:N", title="org unit"), "value:Q"],
            )
            .transform_lookup(
                lookup="id",  # Assuming geojson has org_unit property
                from_=alt.LookupData(agg_df, "org_unit", ["value"]),
            )
            .project(type="equirectangular")  # Use equirectangular projection for proper proportions
            .properties(width=300, height=230, title=title)
        )
        return chart


def make_plot_from_backtest_object(
    backtest: BackTest, plotting_class: MetricPlotV2, metric: MetricBase, geojson=None
) -> alt.Chart:
    # Convert to flat representation using Evaluation abstraction
    evaluation = Evaluation.from_backtest(backtest)
    flat_data = evaluation.to_flat()
    metric_data = metric.compute(flat_data.observations, flat_data.forecasts)
    return plotting_class(metric_data, geojson).plot_spec()
