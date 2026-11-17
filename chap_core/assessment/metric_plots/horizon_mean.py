from typing import cast

import altair as alt

from chap_core.assessment.metric_plots import MetricPlotBase, metric_plot
from chap_core.i18n import N_, gettext


@metric_plot(
    plot_id="metric_by_horizon_mean",
    name=N_("Horizon Plot"),
    description=N_("Shows the aggregated metric by forecast horizon"),
    required_dimensions=["horizon_distance"],
)
class MetricByHorizonV2Mean(MetricPlotBase):
    def plot_from_df(self, title: str = N_("Mean metric by horizon")) -> alt.Chart:
        df = self._metric_data
        adf = df.groupby(["horizon_distance"]).agg({"metric": "mean"}).reset_index()
        return cast(
            "alt.Chart",
            alt.Chart(adf)
            .mark_bar(point=True)
            .encode(
                x=alt.X("horizon_distance:O", title=gettext("Horizon (periods ahead)")),
                y=alt.Y("metric:Q", title=gettext("Mean Metric Value")),
                tooltip=["horizon_distance", "metric"],
            )
            .properties(width=300, height=230, title=gettext(title)),
        )
