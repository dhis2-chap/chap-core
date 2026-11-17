from typing import cast

import altair as alt

from chap_core.assessment.metric_plots import MetricPlotBase, metric_plot
from chap_core.i18n import N_, gettext


@metric_plot(
    plot_id="metric_by_horizon_and_location_mean",
    name=N_("Horizon and Location Plot"),
    description=N_("Shows the aggregated metric by both forecast horizon and location"),
    required_dimensions=["horizon_distance", "location"],
)
class MetricByHorizonAndLocationMean(MetricPlotBase):
    """Mean metric grouped by both horizon distance and location."""

    def plot_from_df(self, title: str = N_("Mean Metric by Horizon and Location")) -> alt.Chart:
        df = self._metric_data
        adf = df.groupby(["horizon_distance", "location"]).agg({"metric": "mean"}).reset_index()
        return cast(
            "alt.Chart",
            alt.Chart(adf)
            .mark_bar(point=True)
            .encode(
                x=alt.X("horizon_distance:O", title=gettext("Horizon (periods ahead)")),
                y=alt.Y("metric:Q", title=gettext("Mean Metric Value")),
                tooltip=["horizon_distance", "location", "metric"],
            )
            .properties(width=300, height=230, title=gettext(title)),
        )
