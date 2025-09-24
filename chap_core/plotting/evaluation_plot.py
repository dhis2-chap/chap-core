import abc
from typing import Optional

import altair as alt
from chap_core.assessment.flat_representations import (
    FlatMetric,
    convert_backtest_observations_to_flat_observations,
    convert_backtest_to_flat_forecasts,
)
from chap_core.assessment.metrics import MetricBase
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

    def plot(self) -> alt.Chart:
        return self.plot_from_df()

    @abc.abstractmethod
    def plot_from_df(self) -> alt.Chart:
        pass

    def plot_spec(self) -> dict:
        chart = self.plot()
        return chart.to_dict()


class VisualizationInfo(DBModel):
    id: str
    display_name: str
    description: str


class MetricByHorizonV2(MetricPlotV2):
    visualization_info = VisualizationInfo(
        id="metric_by_horizon",
        display_name="Horizon Plot",
        description="Shows the aggregated metric by forecast horizon",
    )

    def plot_from_df(self):
        df = self._metric_data
        adf = df.groupby(["horizon_distance", "location"]).agg({"metric": "mean"}).reset_index()
        chart = (
            alt.Chart(adf)
            .mark_bar(point=True)
            .encode(
                x=alt.X("horizon_distance:O", title="Horizon (periods ahead)"),
                y=alt.Y("metric:Q", title="Mean Metric Value"),
                tooltip=["horizon_distance", "location", "metric"],
            )
            .properties(width=600, height=400, title="Mean Metric by Horizon")
            .interactive()
        )

        return chart


class MetricMapV2(MetricPlotV2):
    visualization_info = VisualizationInfo(
        id="metric_map", display_name="Map", description="Shows a map of aggregated metrics per org unit"
    )

    def __init__(self, metric_data: FlatMetric, geojson: Optional[dict] = None):
        super().__init__(metric_data, geojson)
        self._geojson = geojson

    def plot_from_df(self) -> alt.Chart:
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
            .properties(width=600, height=400, title="Metric Map by Location")
        )
        return chart


def make_plot_from_backtest_object(
    backtest: BackTest, plotting_class: MetricPlotV2, metric: MetricBase, geojson=None
) -> alt.Chart:
    # Convert to flat representation
    flat_forecasts = convert_backtest_to_flat_forecasts(backtest.forecasts)
    flat_observations = convert_backtest_observations_to_flat_observations(backtest.dataset.observations)
    metric_data = metric.compute(flat_observations, flat_forecasts)
    return plotting_class(metric_data, geojson).plot_spec()
