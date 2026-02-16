"""
Evaluation plot for backtests.

This module provides a backtest plot that shows truth vs predictions over time,
with uncertainty bands and optional historical observations for context.
"""

from typing import Optional

import altair as alt
import pandas as pd

from chap_core.assessment.backtest_plots import BacktestPlotBase, ChartType, backtest_plot
from chap_core.plotting.backtest_plot import clean_time


def _compute_quantiles_from_forecasts(forecasts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute forecast quantiles from samples.

    Parameters
    ----------
    forecasts_df : pd.DataFrame
        Forecast samples with columns: location, time_period, horizon_distance, sample, forecast

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: location, time_period, split_period, q_10, q_25, q_50, q_75, q_90
    """
    # Group by location, time_period
    # For split_period, we need to calculate it from time_period and horizon_distance
    # split_period = time_period - horizon_distance (conceptually)
    # We'll compute quantiles first, then figure out split_period

    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]

    rows = []
    for (location, time_period, horizon_distance), group in forecasts_df.groupby(
        ["location", "time_period", "horizon_distance"]
    ):
        samples = group["forecast"].values
        quantile_values = pd.Series(samples).quantile(quantiles).values

        # For simplicity, we'll use the first forecast's split calculation
        # In practice, we use horizon_distance to determine the split
        rows.append(
            {
                "time_period": time_period,
                "location": location,
                "horizon_distance": horizon_distance,
                "q_10": quantile_values[0],
                "q_25": quantile_values[1],
                "q_50": quantile_values[2],
                "q_75": quantile_values[3],
                "q_90": quantile_values[4],
            }
        )

    return pd.DataFrame(rows)


def _infer_split_periods(forecasts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Infer split periods from forecasts data.

    For each unique (location, time_period), find the minimum horizon distance
    and use that to determine the split period.

    Parameters
    ----------
    forecasts_df : pd.DataFrame
        DataFrame with columns including location, time_period, horizon_distance

    Returns
    -------
    pd.DataFrame
        DataFrame with added split_period column
    """
    from chap_core.time_period import TimePeriod

    result_rows = []
    for _, row in forecasts_df.iterrows():
        time_period = TimePeriod.parse(str(row["time_period"]))
        horizon = int(row["horizon_distance"])
        # Go back horizon periods to get the split period
        split_period = time_period - (horizon * time_period.time_delta)
        row_dict = row.to_dict()
        row_dict["time_period"] = clean_time(time_period.to_string())
        row_dict["split_period"] = clean_time(split_period.to_string())
        result_rows.append(row_dict)

    return pd.DataFrame(result_rows)


@backtest_plot(
    id="evaluation_plot",
    name="Evaluation Plot",
    description="Shows truth vs predictions over time with uncertainty bands and historical context.",
    needs_historical=True,
)
class EvaluationPlot(BacktestPlotBase):
    """
    Backtest plot that shows truth vs predictions over time.

    Shows forecasts with uncertainty bands and observed values.
    Optionally includes historical observations for context.
    """

    def plot(
        self,
        observations: pd.DataFrame,
        forecasts: pd.DataFrame,
        historical_observations: Optional[pd.DataFrame] = None,
    ) -> ChartType:
        """
        Generate and return the evaluation visualization.

        Parameters
        ----------
        observations : pd.DataFrame
            Observed values with columns: location, time_period, disease_cases
        forecasts : pd.DataFrame
            Forecast samples with columns: location, time_period, horizon_distance,
            sample, forecast
        historical_observations : pd.DataFrame, optional
            Historical observations before split periods, with columns:
            location, time_period, disease_cases

        Returns
        -------
        ChartType
            Altair faceted chart showing forecasts vs observations
        """
        # Compute quantiles from forecast samples
        forecast_quantiles = _compute_quantiles_from_forecasts(forecasts)

        # Add split_period to forecasts
        forecast_df = _infer_split_periods(forecast_quantiles)

        # Clean time periods in observations
        observed_df = observations.copy()
        observed_df["time_period"] = observed_df["time_period"].apply(clean_time)

        # Get unique split periods
        unique_split_periods = forecast_df["split_period"].unique()

        # Create observed data (test periods) with all combinations of split_period
        observed_replicated = []
        for split_period in unique_split_periods:
            tmp = observed_df.copy()
            tmp["split_period"] = split_period
            observed_replicated.append(tmp)

        observed_with_split = pd.concat(observed_replicated, ignore_index=True)

        # Add historical observations if available
        if historical_observations is not None and not historical_observations.empty:
            historical_df = historical_observations.copy()
            historical_df["time_period"] = historical_df["time_period"].apply(clean_time)

            historical_replicated = []
            for split_period in unique_split_periods:
                tmp = historical_df.copy()
                tmp["split_period"] = split_period
                # Filter to only include observations before or at the split_period
                tmp = tmp[tmp["time_period"] <= split_period]
                historical_replicated.append(tmp)

            historical_with_split = pd.concat(historical_replicated, ignore_index=True)

            # Combine historical and test observations
            all_observations = pd.concat(
                [historical_with_split, observed_with_split], ignore_index=True
            ).drop_duplicates(subset=["location", "time_period", "split_period"])
        else:
            all_observations = observed_with_split

        # Prepare data for combined visualization
        forecast_data = forecast_df.copy()
        forecast_data["data_type"] = "forecast"

        observed_data = all_observations.copy()
        observed_data["data_type"] = "observed"

        # Align column names
        for col in ["q_10", "q_25", "q_50", "q_75", "q_90"]:
            if col not in observed_data.columns:
                observed_data[col] = None

        if "disease_cases" not in forecast_data.columns:
            forecast_data["disease_cases"] = None

        # Drop all-NA columns before concatenation
        forecast_data = forecast_data.dropna(axis=1, how="all")
        observed_data = observed_data.dropna(axis=1, how="all")

        # Combine datasets
        combined_data = pd.concat([forecast_data, observed_data], ignore_index=True)

        # Create base chart with combined data
        base = alt.Chart(combined_data)

        # Forecast line (median)
        line = (
            base.transform_filter(alt.datum.data_type == "forecast")
            .mark_line()
            .encode(
                x="time_period:T",
                y=alt.Y("q_50:Q", scale=alt.Scale(zero=False)),
            )
        )

        # Error bands
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

        # Observations (including historical context if available)
        observations_layer = (
            base.transform_filter(alt.datum.data_type == "observed")
            .mark_line(color="orange")
            .encode(
                x="time_period:T",
                y=alt.Y("disease_cases:Q", scale=alt.Scale(zero=False)),
            )
        )

        # Layer all components
        full_layer = error1 + error2 + line + observations_layer

        # Facet the combined layer
        return (  # type: ignore[no-any-return]
            full_layer.facet(column="split_period:O", row="location:N")
            .resolve_scale(y="independent")
            .properties(title="BackTest Forecasts with Observations")
        )
