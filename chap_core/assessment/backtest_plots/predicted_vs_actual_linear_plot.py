"""
Predicted vs actual scatter plot in linear space with regression line.

Shows a scatter plot of predicted (median) vs observed values in linear space
with a line of best fit (OLS regression). Points are colored by location.

Inspired by the Uganda Nutrition Early Warning tool's "Predicted vs Actual —
Walk-Forward CV" chart (CLIM-538).
"""

import altair as alt
import pandas as pd

from chap_core.assessment.backtest_plots import BacktestPlotBase, ChartType, backtest_plot


@backtest_plot(
    plot_id="predicted_vs_actual_linear",
    name="Predicted vs Actual (linear)",
    description=(
        "Scatter plot of predicted (median) vs actual values in linear space "
        "with a line of best fit (regression line), colored by location."
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
        # Compute median forecast per (location, time_period) across all horizons
        median_forecasts = (
            forecasts.groupby(["location", "time_period"]).agg(median_forecast=("forecast", "median")).reset_index()
        )

        merged = median_forecasts.merge(observations, on=["location", "time_period"], how="inner")

        axis_max = max(merged["median_forecast"].max(), merged["disease_cases"].max()) * 1.05

        scatter = (
            alt.Chart(merged)
            .mark_circle(size=60, opacity=0.7)
            .encode(
                x=alt.X(
                    "disease_cases:Q",
                    title="Actual",
                    scale=alt.Scale(domain=[0, axis_max]),
                ),
                y=alt.Y(
                    "median_forecast:Q",
                    title="Predicted",
                    scale=alt.Scale(domain=[0, axis_max]),
                ),
                color=alt.Color("location:N", title="Location"),
                tooltip=[
                    alt.Tooltip("location:N"),
                    alt.Tooltip("time_period:N"),
                    alt.Tooltip("median_forecast:Q", format=".1f", title="Predicted"),
                    alt.Tooltip("disease_cases:Q", format=".1f", title="Actual"),
                ],
            )
        )

        # Line of best fit (OLS regression)
        regression_line = (
            alt.Chart(merged)
            .transform_regression("disease_cases", "median_forecast")
            .mark_line(color="black", strokeDash=[4, 4])
            .encode(
                x="disease_cases:Q",
                y="median_forecast:Q",
            )
        )

        chart = alt.layer(scatter, regression_line).properties(
            width=400, height=400, title="Predicted vs Actual (linear scale)"
        )

        return chart  # type: ignore[no-any-return]
