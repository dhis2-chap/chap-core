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
) -> pd.DataFrame:
    """Aggregate forecast samples to per-horizon medians and inner-join with observed disease cases."""
    median = (
        forecasts.groupby(["location", "time_period", "horizon_distance"])
        .agg(median_forecast=("forecast", "median"))
        .reset_index()
    )
    return median.merge(observations, on=["location", "time_period"], how="inner")


def build_predicted_vs_actual_chart(
    merged: pd.DataFrame,
    *,
    x_field: str,
    y_field: str,
    title: str,
    axis: alt.Axis | None = None,
) -> ChartType:
    """Build a horizon-faceted scatter of predicted vs actual with a dashed identity line."""
    merged = merged.copy()
    merged["_identity_min"] = min(merged[x_field].min(), merged[y_field].min())
    merged["_identity_max"] = max(merged[x_field].max(), merged[y_field].max())

    x_kwargs: dict = {"title": "Predicted"}
    y_kwargs: dict = {"title": "Actual"}
    if axis is not None:
        x_kwargs["axis"] = axis
        y_kwargs["axis"] = axis

    scatter = (
        alt.Chart()
        .mark_circle(size=60, opacity=0.7)
        .encode(
            x=alt.X(f"{x_field}:Q", **x_kwargs),
            y=alt.Y(f"{y_field}:Q", **y_kwargs),
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

    return (  # type: ignore[no-any-return]
        alt.layer(scatter, identity_line, data=merged)
        .properties(width=250, height=250, title=title)
        .facet(column=alt.Column("horizon_distance:O", title="Prediction Horizon"))
    )


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
        merged = median_forecasts_joined_with_observations(forecasts, observations)
        merged["log1p_predicted"] = np.log1p(merged["median_forecast"])
        merged["log1p_actual"] = np.log1p(merged["disease_cases"])

        data_max = max(merged["median_forecast"].max(), merged["disease_cases"].max())
        log1p_ticks = np.log1p(_nice_tick_values(data_max)).tolist()
        log_axis = alt.Axis(values=log1p_ticks, labelExpr="round(exp(datum.value) - 1)")

        return build_predicted_vs_actual_chart(
            merged,
            x_field="log1p_predicted",
            y_field="log1p_actual",
            title="Predicted vs Actual (log1p scale)",
            axis=log_axis,
        )
