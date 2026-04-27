"""
Predicted vs actual scatter plot in linear space.

Each point is one (location, time_period) pair: actual disease cases on the
x-axis, the median forecast on the y-axis, colored by location. A black
dashed identity line (y = x) acts as a visual reference for perfect
prediction — points above the line indicate over-prediction, points below
indicate under-prediction. Linear axes (no log transform) make over- and
under-prediction at high case counts immediately visible.
"""

import altair as alt
import pandas as pd

from chap_core.assessment.backtest_plots import BacktestPlotBase, ChartType, backtest_plot
from chap_core.assessment.backtest_plots.predicted_vs_actual_plot import (
    median_forecasts_joined_with_observations,
)


@backtest_plot(
    plot_id="predicted_vs_actual_linear",
    name="Predicted vs Actual (linear)",
    description=(
        "Scatter of predicted (median) vs actual values in linear space, "
        "colored by location, with a dashed identity reference line."
    ),
)
class PredictedVsActualLinearPlot(BacktestPlotBase):
    def plot(
        self,
        observations: pd.DataFrame,
        forecasts: pd.DataFrame,
        historical_observations: pd.DataFrame | None = None,
        covariates: pd.DataFrame | None = None,
    ) -> ChartType:
        merged = median_forecasts_joined_with_observations(forecasts, observations, by_horizon=False)

        axis_max = max(merged["median_forecast"].max(), merged["disease_cases"].max()) * 1.05

        scatter = (
            alt.Chart(merged)
            .mark_circle(size=60, opacity=0.7)
            .encode(
                x=alt.X("disease_cases:Q", title="Actual", scale=alt.Scale(domain=[0, axis_max])),
                y=alt.Y("median_forecast:Q", title="Predicted", scale=alt.Scale(domain=[0, axis_max])),
                color=alt.Color("location:N", title="Location"),
                tooltip=[
                    alt.Tooltip("location:N"),
                    alt.Tooltip("time_period:N"),
                    alt.Tooltip("disease_cases:Q", format=".0f", title="Actual"),
                    alt.Tooltip("median_forecast:Q", format=".0f", title="Predicted"),
                ],
            )
        )

        identity_line = (
            alt.Chart(pd.DataFrame({"v": [0, axis_max]}))
            .mark_line(color="black", strokeDash=[4, 4])
            .encode(x="v:Q", y="v:Q")
        )

        chart = alt.layer(scatter, identity_line).properties(
            width="container", height=500, title="Predicted vs Actual"
        )

        return chart  # type: ignore[no-any-return]
