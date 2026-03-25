"""
Predicted vs actual scatter plot for backtests.

Shows scatter plots of predicted (median) vs observed values in log1p space,
faceted by prediction horizon and colored by location.
"""

import altair as alt
import numpy as np
import pandas as pd

from chap_core.assessment.backtest_plots import BacktestPlotBase, ChartType, backtest_plot


@backtest_plot(
    plot_id="predicted_vs_actual",
    name="Predicted vs Actual",
    description="Scatter plots of predicted (median) vs actual values in log1p space, faceted by horizon and colored by location.",
)
class PredictedVsActualPlot(BacktestPlotBase):
    def plot(
        self,
        observations: pd.DataFrame,
        forecasts: pd.DataFrame,
        historical_observations: pd.DataFrame | None = None,
    ) -> ChartType:
        median_forecasts = (
            forecasts.groupby(["location", "time_period", "horizon_distance"])
            .agg(median_forecast=("forecast", "median"))
            .reset_index()
        )

        merged = median_forecasts.merge(observations, on=["location", "time_period"], how="inner")
        merged["log1p_predicted"] = np.log1p(merged["median_forecast"])
        merged["log1p_actual"] = np.log1p(merged["disease_cases"])

        chart = (
            alt.Chart(merged)
            .mark_circle(size=60, opacity=0.7)
            .encode(
                x=alt.X("log1p_predicted:Q", title="Predicted (log1p)"),
                y=alt.Y("log1p_actual:Q", title="Actual (log1p)"),
                color=alt.Color("location:N", title="Location"),
                tooltip=[
                    alt.Tooltip("location:N"),
                    alt.Tooltip("time_period:N"),
                    alt.Tooltip("median_forecast:Q", format=".1f", title="Predicted"),
                    alt.Tooltip("disease_cases:Q", format=".1f", title="Actual"),
                ],
            )
            .properties(width=250, height=250, title="Predicted vs Actual (log1p scale)")
            .facet(column=alt.Column("horizon_distance:O", title="Prediction Horizon"))
        )

        return chart  # type: ignore[no-any-return]
