from typing import cast

import altair as alt
import pandas as pd

from chap_core.assessment.metric_plots import MetricPlotBase, metric_plot


@metric_plot(
    plot_id="metric_map",
    name="Map",
    description="Shows a map of aggregated metrics per org unit",
)
class MetricMapV2(MetricPlotBase):
    """Geospatial visualization of aggregated metric per location."""

    def __init__(self, metric_data: pd.DataFrame, geojson: dict | None = None):
        super().__init__(metric_data, geojson)
        self._geojson = geojson

    def plot_from_df(self, title: str = "Metric Map by location") -> alt.Chart:
        df = self._metric_data
        agg_df = df.groupby("location").agg({"metric": "mean"}).reset_index()
        agg_df.rename(columns={"location": "org_unit", "metric": "value"}, inplace=True)

        if self._geojson is None:
            raise ValueError("geojson is required for MetricMapV2")

        return cast(
            "alt.Chart",
            alt.Chart(alt.Data(values=self._geojson["features"]))
            .mark_geoshape(stroke="black", strokeWidth=0.5)
            .encode(
                color=alt.Color("value:Q", scale=alt.Scale(scheme="reds"), title="Metric Value"),
                tooltip=[alt.Tooltip("properties.name:N", title="org unit"), "value:Q"],
            )
            .transform_lookup(
                lookup="id",
                from_=alt.LookupData(agg_df, "org_unit", ["value"]),
            )
            .project(type="equirectangular")
            .properties(width=300, height=230, title=title),
        )
