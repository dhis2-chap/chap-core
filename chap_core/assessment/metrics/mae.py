"""
Mean Absolute Error (MAE) metric.
"""

import pandas as pd
from chap_core.assessment.flat_representations import DataDimension, FlatForecasts, FlatObserved
from chap_core.assessment.metrics.base import MetricBase, MetricSpec


class MAE(MetricBase):
    """
    Mean Absolute Error metric.
    Groups by location and horizon_distance to show error patterns across forecast horizons.
    """

    spec = MetricSpec(output_dimensions=(DataDimension.location, DataDimension.horizon_distance), metric_name="MAE")

    def compute(self, observations: pd.DataFrame, forecasts: pd.DataFrame) -> pd.DataFrame:
        # Compute median forecast across samples for each location/time_period/horizon combination
        median_forecasts = forecasts.groupby(["location", "time_period", "horizon_distance"], as_index=False)[
            "forecast"
        ].median()

        # Merge observations with median forecasts
        merged = median_forecasts.merge(
            observations[["location", "time_period", "disease_cases"]], on=["location", "time_period"], how="inner"
        )

        # Calculate absolute error from median prediction
        merged["abs_error"] = (merged["forecast"] - merged["disease_cases"]).abs()

        # Average across time periods to get MAE per location and horizon
        mae_by_horizon = (
            merged.groupby(["location", "horizon_distance"], as_index=False)["abs_error"]
            .mean()
            .rename(columns={"abs_error": "metric"})
        )

        return mae_by_horizon


class MAEAggregate(MetricBase):
    """
    Fully aggregated Mean Absolute Error metric.
    Computes a single MAE value across all locations, time periods, and horizons.
    Aggregates directly from all data points, not by averaging per-location MAEs.
    """

    spec = MetricSpec(
        output_dimensions=(),
        metric_name="MAE",
        metric_id="mae_aggregate",
        description="Aggregate MAE across all data",
    )

    def compute(self, observations: FlatObserved, forecasts: FlatForecasts) -> pd.DataFrame:
        # Compute median forecast across samples for each location/time_period/horizon combination
        median_forecasts = forecasts.groupby(["location", "time_period", "horizon_distance"], as_index=False)[
            "forecast"
        ].median()

        # Merge observations with median forecasts
        merged = median_forecasts.merge(
            observations[["location", "time_period", "disease_cases"]], on=["location", "time_period"], how="inner"
        )

        # Calculate absolute error from median prediction
        merged["abs_error"] = (merged["forecast"] - merged["disease_cases"]).abs()

        # Average absolute error across all entries
        mae = merged["abs_error"].mean()

        return pd.DataFrame({"metric": [mae]})
