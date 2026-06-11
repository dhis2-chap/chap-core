from typing import cast

import altair as alt

from chap_core.assessment.metric_plots import MetricPlotBase, metric_plot


@metric_plot(
    plot_id="metric_by_time_period_and_location_mean",
    name="Time Period and Location Plot",
    description="Shows the aggregated metric by both time period and location",
)
class MetricByTimePeriodAndLocationV2Mean(MetricPlotBase):
    """Mean metric grouped by time period and location."""

    def plot_from_df(self, title: str = "Mean metric by location and time period") -> alt.Chart:
        df = self._metric_data
        adf = df.groupby(["time_period", "location"]).agg({"metric": "mean"}).reset_index()
        return cast(
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
