from typing import cast

import altair as alt

from chap_core.assessment.metric_plots import MetricPlotBase, metric_plot
from chap_core.i18n import N_, gettext


@metric_plot(
    plot_id="metric_by_horizon_sum",
    name=N_("Horizon Plot (Sum)"),
    description=N_("Shows the sum of the metric across locations for each forecast horizon"),
    required_dimensions=["horizon_distance", "location"],
)
class MetricByHorizonV2Sum(MetricPlotBase):
    """Sum of metric across locations per forecast horizon."""

    def plot_from_df(self, title: str = N_("Samples above truth by horizon")) -> alt.Chart:
        df = self._metric_data
        adf = df.groupby(["horizon_distance"]).agg({"metric": "sum"}).reset_index()
        return cast(
            "alt.Chart",
            alt.Chart(adf)
            .mark_bar()
            .encode(
                x=alt.X("horizon_distance:O", title=gettext("Horizon (periods ahead)")),
                y=alt.Y("metric:Q", title=gettext("Samples above truth (count)")),
                tooltip=[
                    alt.Tooltip("horizon_distance:O", title=gettext("Horizon")),
                    alt.Tooltip("metric:Q", title=gettext("Count")),
                ],
            )
            .properties(width=300, height=230, title=gettext(title)),
        )
