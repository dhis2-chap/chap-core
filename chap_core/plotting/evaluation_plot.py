import abc
from typing import cast

import altair as alt
import pandas as pd

from chap_core.assessment.evaluation import Evaluation
from chap_core.assessment.metrics.base import Metric
from chap_core.database.base_tables import DBModel
from chap_core.database.tables import BackTest

alt.renderers.enable("browser")


class MetricPlotV2(abc.ABC):
    """
    Represents types of metrics plots, that always start from raw FlatMetric data.
    Differnet plots can process this data in the way they want to produce a plot
    """

    visualization_info: "VisualizationInfo"  # Declared by subclasses

    def __init__(self, metric_data: pd.DataFrame, geojson: dict | None = None):
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

    def plot_from_df(self, title: str = "Mean Metric by Horizon") -> alt.Chart:
        df = self._metric_data
        adf = df.groupby(["horizon_distance", "location"]).agg({"metric": "mean"}).reset_index()
        print(adf)
        chart = cast(
            "alt.Chart",
            alt.Chart(adf)
            .mark_bar(point=True)
            .encode(
                x=alt.X("horizon_distance:O", title="Horizon (periods ahead)"),
                y=alt.Y("metric:Q", title="Mean Metric Value"),
                tooltip=["horizon_distance", "location", "metric"],
            )
            .properties(width=300, height=230, title=title),
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

    def plot_from_df(self, title: str = "Samples above truth by horizon") -> alt.Chart:
        df = self._metric_data
        chart = cast(
            "alt.Chart",
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
            .properties(width=300, height=230, title=title),
        )

        return chart


class MetricByTimePeriodAndLocationV2Mean(MetricPlotV2):
    visualization_info = VisualizationInfo(
        id="metric_by_time_period",
        display_name="Time Period Plot",
        description="Shows the aggregated metric by time period (per location)",
    )

    def plot_from_df(self, title: str = "Mean metric by location and time period") -> alt.Chart:
        df = self._metric_data
        adf = df.groupby(["time_period", "location"]).agg({"metric": "mean"}).reset_index()
        chart = cast(
            "alt.Chart",
            alt.Chart(adf)
            .mark_line(point=True)
            .encode(
                x=alt.X("time_period:O", title="Time period"),
                y=alt.Y("metric:Q", title="Mean Metric Value"),
                color=alt.Color("location:N", title="Location"),
                tooltip=["time_period", "location", "metric"],
            )
            .properties(width=300, height=230, title=title),
        )

        return chart


class MetricByTimePeriodV2Sum(MetricPlotV2):
    visualization_info = VisualizationInfo(
        id="metric_by_time_sum",
        display_name="Horizon Plot (sum)",
        description="Sums metric across locations per forecast horizon",
    )

    def plot_from_df(self, title: str = "Samples above truth by time period") -> alt.Chart:
        df = self._metric_data
        chart = cast(
            "alt.Chart",
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
            .properties(width=300, height=230, title=title),
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


class RegionalMetricDistributionPlot(MetricPlotV2):
    """Backtest plot showing location-level error distributions with mean markers."""

    visualization_info = VisualizationInfo(
        id="regional_metric_distribution",
        display_name="Regional Metric Distribution",
        description="Boxplots showing Metric distributions by location with mean points",
    )

    def plot_from_df(self, title: str = "Metric Distribution by Location") -> alt.Chart:
        df = self._metric_data

        if df.empty:
            return (  # type: ignore[no-any-return]
                alt.Chart(pd.DataFrame({"message": ["No valid rows available for boxplot statistics"]}))
                .mark_text(align="left", fontSize=14)
                .encode(text="message:N")
                .properties(height=60)
            )

        by_location = (
            alt.Chart(df)
            .mark_boxplot(extent=1.5, size=30, median={"stroke": "royalblue"})
            .encode(
                x=alt.X("location:N", title="Location"),
                y=alt.Y("metric:Q", title="Metric"),
                tooltip=["location:N", "time_period:N", "horizon_distance:O", alt.Tooltip("metric:Q", format=".2f")],
            )
            .properties(
                width=700,
                height=280,
                title=title,
            )
        )

        mean_by_location = df.groupby("location", as_index=False).agg(mean_metric=("metric", "mean"))

        mean_points = (
            alt.Chart(mean_by_location)
            .mark_point(color="#F58518", size=50, filled=True)
            .encode(
                x=alt.X("location:N", title="Location"),
                y=alt.Y("mean_metric:Q", title="Metric"),
                tooltip=["location:N", alt.Tooltip("mean_metric:Q", format=".2f", title="Mean Metric")],
            )
        )

        chart = by_location + mean_points

        return (  # type: ignore[no-any-return]
            chart.configure_axis(labelFontSize=11, titleFontSize=12)
            .configure_legend(labelFontSize=11, titleFontSize=12)
            .configure_view(stroke=None)
            .configure_mark(opacity=0.5, color="royalblue")
        )


class MetricMapV2(MetricPlotV2):
    visualization_info = VisualizationInfo(
        id="metric_map", display_name="Map", description="Shows a map of aggregated metrics per org unit"
    )

    def __init__(self, metric_data: pd.DataFrame, geojson: dict | None = None):
        super().__init__(metric_data, geojson)
        self._geojson = geojson

    def plot_from_df(self, title: str = "Metric Map by location") -> alt.Chart:
        # Get the metric data DataFrame
        df = self._metric_data

        # Aggregate metrics by location (average across all time periods and horizons)
        agg_df = df.groupby("location").agg({"metric": "mean"}).reset_index()
        agg_df.rename(columns={"location": "org_unit", "metric": "value"}, inplace=True)

        # Create map visualization with geojson
        geojson_data = self._geojson
        if geojson_data is None:
            raise ValueError("geojson is required for MetricMapV2")

        # Build Altair map chart
        chart = cast(
            "alt.Chart",
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
            .properties(width=300, height=230, title=title),
        )
        return chart


def make_plot_from_backtest_object(
    backtest: BackTest, plotting_class: type[MetricPlotV2], metric: Metric, geojson: dict | None = None
) -> dict:
    # Convert to flat representation using Evaluation abstraction
    evaluation = Evaluation.from_backtest(backtest)
    flat_data = evaluation.to_flat()
    metric_data = metric.get_detailed_metric(flat_data.observations, flat_data.forecasts)
    return plotting_class(metric_data, geojson).plot_spec()
