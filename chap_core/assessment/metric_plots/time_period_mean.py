from typing import cast

import altair as alt

from chap_core.assessment.metric_plots import MetricPlotBase, metric_plot


@metric_plot(
    plot_id="metric_by_time_period_mean",
    name="Time Period Plot",
    description="Shows the aggregated metric by time period",
)
class MetricByTimePeriodV2Mean(MetricPlotBase):
    """Mean metric across locations and horizons per time period."""

    def plot_from_df(self, title: str = "Mean metric by time period") -> alt.Chart:
        df = self._metric_data
        df = df.groupby(["time_period"]).agg({"metric": "mean"}).reset_index()
        return cast(
            "alt.Chart",
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
            .properties(width=300, height=230, title=title),
        )
