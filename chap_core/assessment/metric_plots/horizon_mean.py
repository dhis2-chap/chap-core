from typing import cast

import altair as alt

from chap_core.assessment.metric_plots import MetricPlotBase, metric_plot


@metric_plot(
    plot_id="metric_by_horizon_mean",
    name="Horizon Plot",
    description="Shows the aggregated metric by forecast horizon",
)
class MetricByHorizonV2Mean(MetricPlotBase):
    def plot_from_df(self, title: str = "Mean metric by horizon") -> alt.Chart:
        df = self._metric_data
        adf = df.groupby(["horizon_distance"]).agg({"metric": "mean"}).reset_index()
        return cast(
            "alt.Chart",
            alt.Chart(adf)
            .mark_bar(point=True)
            .encode(
                x=alt.X("horizon_distance:O", title="Horizon (periods ahead)"),
                y=alt.Y("metric:Q", title="Mean Metric Value"),
                tooltip=["horizon_distance", "metric"],
            )
            .properties(width=300, height=230, title=title),
        )
