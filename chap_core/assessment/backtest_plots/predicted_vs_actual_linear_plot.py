"""
Predicted vs actual scatter plot in linear space with regression line.

Shows a scatter plot of predicted (median) vs observed values in linear space
with a line of best fit (OLS regression). Inspired by the Uganda Nutrition
Early Warning tool's "Predicted vs Actual" chart (CLIM-538).
"""

import altair as alt
import pandas as pd

from chap_core.assessment.backtest_plots import BacktestPlotBase, ChartType, backtest_plot


@backtest_plot(
    plot_id="predicted_vs_actual_linear",
    name="Predicted vs Actual (linear)",
    description=(
        "Scatter plot of predicted (median) vs actual values in linear space with a line of best fit (regression line)."
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
        median_forecasts = (
            forecasts.groupby(["location", "time_period"]).agg(median_forecast=("forecast", "median")).reset_index()
        )

        merged = median_forecasts.merge(observations, on=["location", "time_period"], how="inner")

        axis_max = max(merged["median_forecast"].max(), merged["disease_cases"].max()) * 1.05

        base = alt.Chart(merged)

        scatter = base.mark_circle(size=40, opacity=0.6, color="steelblue").encode(
            x=alt.X("disease_cases:Q", title="Actual", scale=alt.Scale(domain=[0, axis_max])),
            y=alt.Y("median_forecast:Q", title="Predicted", scale=alt.Scale(domain=[0, axis_max])),
            tooltip=[
                alt.Tooltip("location:N"),
                alt.Tooltip("time_period:N"),
                alt.Tooltip("disease_cases:Q", format=".0f", title="Actual"),
                alt.Tooltip("median_forecast:Q", format=".0f", title="Predicted"),
            ],
        )

        regression_line = (
            base.transform_regression("disease_cases", "median_forecast")
            .mark_line(color="black", strokeDash=[6, 3], strokeWidth=1.5)
            .encode(x="disease_cases:Q", y="median_forecast:Q")
        )

        identity_line = (
            alt.Chart(pd.DataFrame({"v": [0, axis_max]}))
            .mark_line(color="gray", strokeDash=[2, 2], opacity=0.4)
            .encode(x="v:Q", y="v:Q")
        )

        chart = alt.layer(scatter, regression_line, identity_line).properties(
            width=350, height=350, title="Predicted vs Actual"
        )

        return chart  # type: ignore[no-any-return]
