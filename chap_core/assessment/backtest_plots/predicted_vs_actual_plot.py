"""
Predicted vs actual scatter plots for backtests.

Shows scatter plots of predicted (median) vs observed values,
faceted by prediction horizon and colored by location.
"""

import altair as alt
import numpy as np
import pandas as pd

from chap_core.assessment.backtest_plots import BacktestPlotBase, ChartType, backtest_plot


def _predicted_vs_actual_chart(
    observations: pd.DataFrame,
    forecasts: pd.DataFrame,
    log1p: bool,
) -> ChartType:
    median_forecasts = (
        forecasts.groupby(["location", "time_period", "horizon_distance"])
        .agg(median_forecast=("forecast", "median"))
        .reset_index()
    )

    merged = median_forecasts.merge(observations, on=["location", "time_period"], how="inner")

    merged["horizon_distance"] = merged["horizon_distance"] + 1
    is_weekly = merged["time_period"].iloc[0].upper().count("W") > 0 if len(merged) > 0 else False
    horizon_unit = "weeks" if is_weekly else "months"

    if log1p:
        merged["plot_predicted"] = np.log1p(merged["median_forecast"])
        merged["plot_actual"] = np.log1p(merged["disease_cases"])
        suffix = " (log1p)"
    else:
        merged["plot_predicted"] = merged["median_forecast"]
        merged["plot_actual"] = merged["disease_cases"]
        suffix = ""

    scale_desc = "log(1+x) transformed" if log1p else "untransformed"
    caption_text = (
        f"Scatter plot of median predicted vs actual disease cases ({scale_desc}). "
        f"Each point represents one location at one time period. "
        f"Points on the diagonal indicate perfect predictions. "
        f"Panels show different prediction horizons in {horizon_unit}."
    )
    n_horizons = len(merged["horizon_distance"].unique())
    caption_width = n_horizons * 250 + (n_horizons - 1) * 10
    import textwrap

    wrapped = textwrap.wrap(caption_text, width=130)
    caption_df = pd.DataFrame({"line": wrapped, "y": range(len(wrapped))})
    caption = (
        alt.Chart(caption_df)
        .mark_text(align="left", baseline="top", fontSize=12, dx=-(caption_width // 2) + 10)
        .encode(text="line:N", y=alt.Y("y:O", axis=None))
        .properties(width=caption_width, height=len(wrapped) * 14 + 20)
    )

    scatter = (
        alt.Chart(merged)
        .mark_circle(size=60, opacity=0.7)
        .encode(
            x=alt.X("plot_predicted:Q", title=f"Predicted{suffix}"),
            y=alt.Y("plot_actual:Q", title=f"Actual{suffix}"),
            color=alt.Color("location:N", title="Location"),
            tooltip=[
                alt.Tooltip("location:N"),
                alt.Tooltip("time_period:N"),
                alt.Tooltip("median_forecast:Q", format=".1f", title="Predicted"),
                alt.Tooltip("disease_cases:Q", format=".1f", title="Actual"),
            ],
        )
        .properties(width=250, height=250, title=f"Predicted vs Actual{suffix}")
        .facet(column=alt.Column("horizon_distance:O", title=f"Prediction Horizon ({horizon_unit})"))
    )

    return alt.vconcat(scatter, caption)  # type: ignore[no-any-return]


@backtest_plot(
    plot_id="predicted_vs_actual",
    name="Predicted vs Actual (log1p)",
    description="Scatter plots of predicted (median) vs actual values in log1p space, faceted by horizon and colored by location.",
)
class PredictedVsActualPlot(BacktestPlotBase):
    def plot(
        self,
        observations: pd.DataFrame,
        forecasts: pd.DataFrame,
        historical_observations: pd.DataFrame | None = None,
    ) -> ChartType:
        return _predicted_vs_actual_chart(observations, forecasts, log1p=True)


@backtest_plot(
    plot_id="predicted_vs_actual_linear",
    name="Predicted vs Actual (linear)",
    description="Scatter plots of predicted (median) vs actual values in linear space, faceted by horizon and colored by location.",
)
class PredictedVsActualLinearPlot(BacktestPlotBase):
    def plot(
        self,
        observations: pd.DataFrame,
        forecasts: pd.DataFrame,
        historical_observations: pd.DataFrame | None = None,
    ) -> ChartType:
        return _predicted_vs_actual_chart(observations, forecasts, log1p=False)
