from typing import cast

import altair as alt

from chap_core.assessment.metric_plots import MetricPlotBase


class MetricByTimePeriodV2Sum(MetricPlotBase):
    """Sum of metric across locations per time period."""

    def plot_from_df(self, title: str = "Samples above truth by time period") -> alt.Chart:
        df = self._metric_data
        return cast(
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
