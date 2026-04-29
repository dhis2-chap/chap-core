"""
Predicted vs actual scatter plot for backtests.

Shows scatter plots of predicted (median) vs observed values in log1p space,
faceted by prediction horizon and colored by location.
"""

import altair as alt
import numpy as np
import pandas as pd

from chap_core.assessment.backtest_plots import BacktestPlotBase, ChartType, backtest_plot


def _nice_tick_values(data_max: float) -> np.ndarray:
    """Generate nice round tick values from 0 up to data_max."""
    candidates = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
    return np.array([v for v in candidates if v <= data_max * 1.1])


def median_forecasts_joined_with_observations(
    forecasts: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    by_horizon: bool,
) -> pd.DataFrame:
    """Aggregate forecast samples to medians and inner-join with observed disease cases.

    by_horizon=True keeps horizon_distance in the groupby (one median per
    location/time_period/horizon — used by the faceted log1p plot).
    by_horizon=False pools across horizon (one median per location/time_period —
    used by the linear scatter).
    """
    group_cols = ["location", "time_period", "horizon_distance"] if by_horizon else ["location", "time_period"]
    median = forecasts.groupby(group_cols).agg(median_forecast=("forecast", "median")).reset_index()
    return median.merge(observations, on=["location", "time_period"], how="inner")


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
        merged = median_forecasts_joined_with_observations(forecasts, observations, by_horizon=True)
        merged["log1p_predicted"] = np.log1p(merged["median_forecast"])
        merged["log1p_actual"] = np.log1p(merged["disease_cases"])

        data_max = max(merged["median_forecast"].max(), merged["disease_cases"].max())
        tick_values = _nice_tick_values(data_max)
        log1p_ticks = np.log1p(tick_values).tolist()

        axis_min = min(merged["log1p_predicted"].min(), merged["log1p_actual"].min())
        axis_max = max(merged["log1p_predicted"].max(), merged["log1p_actual"].max())
        merged["_identity_min"] = axis_min
        merged["_identity_max"] = axis_max

        scatter = (
            alt.Chart()
            .mark_circle(size=60, opacity=0.7)
            .encode(
                x=alt.X(
                    "log1p_predicted:Q",
                    title="Predicted",
                    axis=alt.Axis(
                        values=log1p_ticks,
                        labelExpr="round(exp(datum.value) - 1)",
                    ),
                ),
                y=alt.Y(
                    "log1p_actual:Q",
                    title="Actual",
                    axis=alt.Axis(
                        values=log1p_ticks,
                        labelExpr="round(exp(datum.value) - 1)",
                    ),
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

        identity_line = (
            alt.Chart()
            .mark_line(color="black", strokeDash=[4, 4])
            .transform_fold(["_identity_min", "_identity_max"], as_=["_key", "_val"])
            .encode(x=alt.X("_val:Q"), y=alt.Y("_val:Q"))
        )

        chart = (
            alt.layer(scatter, identity_line, data=merged)
            .properties(width=250, height=250, title="Predicted vs Actual (log1p scale)")
            .facet(column=alt.Column("horizon_distance:O", title="Prediction Horizon"))
        )

        return chart  # type: ignore[no-any-return]
