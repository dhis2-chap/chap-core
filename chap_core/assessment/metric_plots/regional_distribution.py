import altair as alt
import pandas as pd

from chap_core.assessment.metric_plots import MetricPlotBase, metric_plot


@metric_plot(
    plot_id="regional_metric_distribution",
    name="Regional Metric Distribution",
    description="Boxplots showing metric distributions by location with mean points",
)
class RegionalMetricDistributionPlot(MetricPlotBase):
    """Location-level metric distributions with mean markers."""

    def plot_from_df(self, title: str = "Metric Distribution by Location") -> alt.Chart:
        df = self._metric_data

        if df.empty:
            return (  # type: ignore[no-any-return]
                alt.Chart(pd.DataFrame({"message": ["No valid rows available for boxplot statistics"]}))
                .mark_text(align="left", fontSize=14)
                .encode(text="message:N")
                .properties(height=60)
            )

        by_location = (
            alt.Chart(df)
            .mark_boxplot(extent=1.5, size=30, median={"stroke": "royalblue"})
            .encode(
                x=alt.X("location:N", title="Location"),
                y=alt.Y("metric:Q", title="Metric"),
                tooltip=[
                    "location:N",
                    "time_period:N",
                    "horizon_distance:O",
                    alt.Tooltip("metric:Q", format=".2f"),
                ],
            )
            .properties(width=700, height=280, title=title)
        )

        mean_by_location = df.groupby("location", as_index=False).agg(mean_metric=("metric", "mean"))

        mean_points = (
            alt.Chart(mean_by_location)
            .mark_point(color="#F58518", size=50, filled=True)
            .encode(
                x=alt.X("location:N", title="Location"),
                y=alt.Y("mean_metric:Q", title="Metric"),
                tooltip=["location:N", alt.Tooltip("mean_metric:Q", format=".2f", title="Mean Metric")],
            )
        )

        return (  # type: ignore[no-any-return]
            (by_location + mean_points)
            .configure_axis(labelFontSize=11, titleFontSize=12)
            .configure_legend(labelFontSize=11, titleFontSize=12)
            .configure_view(stroke=None)
            .configure_mark(opacity=0.5, color="royalblue")
        )
