"""Backtest boxplot statistics visualization.

Shows distribution of forecast errors as boxplots by location.
"""

import altair as alt
import pandas as pd

from chap_core.assessment.backtest_plots import BacktestPlotBase, ChartType, backtest_plot
from chap_core.assessment.flat_representations import FlatForecasts, FlatObserved
from chap_core.assessment.metrics import MAPEMetric


def _compute_detailed_mape(observations: pd.DataFrame, forecasts: pd.DataFrame) -> pd.DataFrame:
    """Compute detailed MAPE metric rows using the predefined metric implementation."""
    metric = MAPEMetric()
    detailed = metric.get_detailed_metric(FlatObserved(observations), FlatForecasts(forecasts))
    detailed["metric_name"] = metric.get_name()
    return detailed


@backtest_plot(
    plot_id="regional_mape_distribution",
    name="Regional MAPE distribution",
    description="Boxplots showing MAPE error distributions by location with mean points.",
)
class RegionalMAPEDistributionPlot(BacktestPlotBase):
    """Backtest plot showing location-level error distributions with mean markers."""

    def plot(
        self,
        observations: pd.DataFrame,
        forecasts: pd.DataFrame,
        historical_observations: pd.DataFrame | None = None,
    ) -> ChartType:
        mape_detailed = _compute_detailed_mape(observations, forecasts)

        if mape_detailed.empty:
            return (  # type: ignore[no-any-return]
                alt.Chart(pd.DataFrame({"message": ["No valid rows available for boxplot statistics"]}))
                .mark_text(align="left", fontSize=14)
                .encode(text="message:N")
                .properties(height=60)
            )

        by_location = (
            alt.Chart(mape_detailed)
            .mark_boxplot(extent=1.5, size=30, median={"stroke": "royalblue"})
            .encode(
                x=alt.X("location:N", title="Location"),
                y=alt.Y("metric:Q", title="MAPE"),
                tooltip=["location:N", "time_period:N", "horizon_distance:O", alt.Tooltip("metric:Q", format=".2f")],
            )
            .properties(
                width=700,
                height=280,
                title="MAPE Distribution by Location",
            )
        )

        mean_by_location = mape_detailed.groupby("location", as_index=False).agg(mean_mape=("metric", "mean"))

        mean_points = (
            alt.Chart(mean_by_location)
            .mark_point(color="#F58518", size=50, filled=True)
            .encode(
                x=alt.X("location:N", title="Location"),
                y=alt.Y("mean_mape:Q", title="MAPE"),
                tooltip=["location:N", alt.Tooltip("mean_mape:Q", format=".2f", title="Mean MAPE")],
            )
        )

        chart = by_location + mean_points

        return (  # type: ignore[no-any-return]
            chart.configure_axis(labelFontSize=11, titleFontSize=12)
            .configure_legend(labelFontSize=11, titleFontSize=12)
            .configure_view(stroke=None)
            .configure_mark(opacity=0.5, color="royalblue")
        )
