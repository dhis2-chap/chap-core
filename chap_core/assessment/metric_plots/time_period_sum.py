from typing import cast

import altair as alt

from chap_core.assessment.metric_plots import MetricPlotBase, metric_plot


@metric_plot(
    plot_id="metric_by_time_period_sum",
    name="Time Period Plot (Sum)",
    description="Shows the sum of the metric by time period",
)
class MetricByTimePeriodV2Sum(MetricPlotBase):
    """Sum of metric across locations per time period."""

    def plot_from_df(self, title: str = "Samples above truth by time period") -> alt.Chart:
        df = self._metric_data
        adf = df.groupby(["time_period", "location"]).agg({"metric": "sum"}).reset_index()
        return cast(
            "alt.Chart",
            alt.Chart(adf)
            .mark_line()
            .encode(
                x=alt.X("time_period:O", title="Time Period"),
                y=alt.Y("metric:Q", title="Samples above truth (count)"),
                color=alt.Color("location:N", title="Location"),
                tooltip=[
                    alt.Tooltip("time_period:O", title="Time Period"),
                    alt.Tooltip("metric:Q", title="Count"),
                ],
            )
            .properties(width=300, height=230, title=title),
        )
