from typing import cast

import altair as alt

from chap_core.assessment.metric_plots import MetricPlotBase, metric_plot
from chap_core.i18n import N_, gettext


@metric_plot(
    plot_id="metric_by_time_period_mean",
    name=N_("Time Period Plot"),
    description=N_("Shows the aggregated metric by time period"),
    required_dimensions=["time_period"],
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
                x=alt.X("time_period:O", title=gettext("Time Period")),
                y=alt.Y("metric:Q", title=gettext("Mean Metric Value")),
                tooltip=[
                    alt.Tooltip("time_period:O", title=gettext("Time Period")),
                    alt.Tooltip("metric:Q", title=gettext("Count")),
                ],
            )
            .properties(width=300, height=230, title=gettext(title)),
        )
