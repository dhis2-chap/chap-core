from typing import cast

import altair as alt

from chap_core.assessment.metric_plots import MetricPlotBase, metric_plot
from chap_core.i18n import N_, gettext


@metric_plot(
    plot_id="metric_by_time_period_sum",
    name=N_("Time Period Plot (Sum)"),
    description=N_("Shows the sum of the metric by time period"),
    required_dimensions=["time_period", "location"],
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
                x=alt.X("time_period:O", title=gettext("Time Period")),
                y=alt.Y("metric:Q", title=gettext("Samples above truth (count)")),
                color=alt.Color("location:N", title=gettext("Location")),
                tooltip=[
                    alt.Tooltip("time_period:O", title=gettext("Time Period")),
                    alt.Tooltip("metric:Q", title=gettext("Count")),
                ],
            )
            .properties(width=300, height=230, title=gettext(title)),
        )
