from typing import cast

import altair as alt

from chap_core.assessment.metric_plots import MetricPlotBase


class MetricByHorizonV2Sum(MetricPlotBase):
    """Sum of metric across locations per forecast horizon."""

    def plot_from_df(self, title: str = "Samples above truth by horizon") -> alt.Chart:
        df = self._metric_data
        return cast(
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
