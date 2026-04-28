from typing import cast

import altair as alt

from chap_core.assessment.metric_plots import MetricPlotBase


class MetricByHorizonAndLocationMean(MetricPlotBase):
    """Mean metric grouped by both horizon distance and location."""

    def plot_from_df(self, title: str = "Mean Metric by Horizon and Location") -> alt.Chart:
        df = self._metric_data
        adf = df.groupby(["horizon_distance", "location"]).agg({"metric": "mean"}).reset_index()
        return cast(
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
