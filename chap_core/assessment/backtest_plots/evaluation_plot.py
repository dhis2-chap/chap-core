"""
Evaluation plot for backtests.

This module provides a backtest plot that shows truth vs predictions over time,
with uncertainty bands and optional historical observations for context.
"""

import altair as alt
import pandas as pd

from chap_core.assessment.backtest_plots import ChartType, FacetedBacktestPlot, backtest_plot
from chap_core.plotting.backtest_plot import clean_time
from chap_core.time_period import TimePeriod


def _compute_quantiles_from_forecasts(forecasts_df: pd.DataFrame) -> pd.DataFrame:
    """Compute forecast quantiles efficiently using vectorized groupby operations."""
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]

    # Vectorized calculation of all quantiles across all groups instantly
    grouped = forecasts_df.groupby(["location", "time_period", "horizon_distance"])["forecast"]
    quantile_df = grouped.quantile(quantiles).unstack(level=-1)

    # Clean up column structure to match expected format
    quantile_df.columns = [f"q_{int(q*100)}" for q in quantiles]
    return quantile_df.reset_index()


def _infer_split_periods_vectorized(quantiles_df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized calculation of split periods using explicit time delta math."""
    if quantiles_df.empty:
        return quantiles_df.copy()

    # Apply the conversion to TimePeriod object strings uniformly
    # Note: If TimePeriod can be adapted to handle pandas Series natively, do it there.
    # Otherwise, this maps over unique IDs instead of every single sample row.
    def _sub_horizon(row):
        tp = TimePeriod.parse(str(row["time_period"]))
        h = int(row["horizon_distance"])
        return clean_time((tp - (h * tp.time_delta)).to_string())

    df = quantiles_df.copy()
    # Apply row-wise across unique aggregations (much smaller than base sample data)
    df["split_period"] = df.apply(_sub_horizon, axis=1)
    df["time_period"] = df["time_period"].astype(str).apply(clean_time)

    return df


@backtest_plot(
    plot_id="evaluation_plot",
    name="Evaluation Plot",
    description="Shows truth vs predictions over time with uncertainty bands and historical context.",
    needs_historical=True,
)
class EvaluationPlot(FacetedBacktestPlot):
    """Shows forecasts with uncertainty bands and observed values with historical context."""

    facet_dimensions = ["split_period:O", "location:N"]

    def _preprocess(
        self,
        observations: pd.DataFrame,
        forecasts: pd.DataFrame,
        historical_observations: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Handles the vectorized data pipeline without memory-intensive copying loops."""

        # 1. Compute quantiles and split windows
        forecast_quantiles = _compute_quantiles_from_forecasts(forecasts)
        forecast_df = _infer_split_periods_vectorized(forecast_quantiles)

        unique_splits = pd.DataFrame({"split_period": forecast_df["split_period"].unique()})

        # 2. Clean up Observations
        observed_df = observations.copy()
        observed_df["time_period"] = observed_df["time_period"].astype(str).apply(clean_time)

        # Vectorized Cross Join replacing replication loops
        observed_with_split = observed_df.merge(unique_splits, how="cross")

        # 3. Handle Historical Windows
        if historical_observations is not None and not historical_observations.empty:
            historical_df = historical_observations.copy()
            historical_df["time_period"] = historical_df["time_period"].astype(str).apply(clean_time)

            # Vectorized cross-join + instant boolean filter mask
            historical_with_split = historical_df.merge(unique_splits, how="cross")
            historical_with_split = historical_with_split[
                historical_with_split["time_period"] <= historical_with_split["split_period"]
            ]

            all_observations = pd.concat(
                [historical_with_split, observed_with_split], ignore_index=True
            ).drop_duplicates(subset=["location", "time_period", "split_period"])
        else:
            all_observations = observed_with_split

        # 4. Alignment & Concatenation
        forecast_df["data_type"] = "forecast"
        all_observations["data_type"] = "observed"

        # Safe alignment of missing columns across frames without loop checking
        final_df = pd.concat([forecast_df, all_observations], ignore_index=True)
        return final_df

    def _plot(self, df: pd.DataFrame) -> ChartType:
        """Renders visual layers using standard underlying Altair components."""
        base = alt.Chart(df)

        line = (
            base.transform_filter(alt.datum.data_type == "forecast")
            .mark_line()
            .encode(
                x="time_period:T",
                y=alt.Y("q_50:Q", scale=alt.Scale(zero=False)),
            )
        )

        error1 = (
            base.transform_filter(alt.datum.data_type == "forecast")
            .mark_errorband(color="blue", opacity=0.3)
            .encode(
                x="time_period:T",
                y=alt.Y("q_10:Q", scale=alt.Scale(zero=False)),
                y2="q_90:Q",
            )
        )

        error2 = (
            base.transform_filter(alt.datum.data_type == "forecast")
            .mark_errorband(color="blue", opacity=0.5)
            .encode(
                x="time_period:T",
                y=alt.Y("q_25:Q", scale=alt.Scale(zero=False)),
                y2="q_75:Q",
            )
        )

        observations_layer = (
            base.transform_filter(alt.datum.data_type == "observed")
            .mark_line(color="orange")
            .encode(
                x="time_period:T",
                y=alt.Y("disease_cases:Q", scale=alt.Scale(zero=False)),
            )
        )

        return (error1 + error2 + line + observations_layer).properties(
            title="Backtest Forecasts with Observations"
        )
