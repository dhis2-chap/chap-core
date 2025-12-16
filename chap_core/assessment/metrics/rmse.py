"""
Root Mean Squared Error (RMSE) metrics.
"""

import pandas as pd
from chap_core.assessment.flat_representations import DataDimension, FlatForecasts, FlatObserved
from chap_core.assessment.metrics.base import MetricBase, MetricSpec


class RMSE(MetricBase):
    """
    Root Mean Squared Error metric.
    Groups by location to give RMSE per location across all time periods and horizons.
    """

    spec = MetricSpec(output_dimensions=(DataDimension.location,), metric_name="RMSE")

    def compute(self, observations: FlatObserved, forecasts: FlatForecasts) -> pd.DataFrame:
        # Compute median forecast across samples for each location/time_period/horizon combination
        median_forecasts = forecasts.groupby(["location", "time_period", "horizon_distance"], as_index=False)[
            "forecast"
        ].median()

        # Merge observations with median forecasts on location and time_period
        merged = median_forecasts.merge(
            observations[["location", "time_period", "disease_cases"]], on=["location", "time_period"], how="inner"
        )

        # Calculate squared error from median prediction
        merged["squared_error"] = (merged["forecast"] - merged["disease_cases"]) ** 2

        # Average across all time periods for each location
        location_mse = merged.groupby("location", as_index=False)["squared_error"].mean()

        # Take square root to get RMSE
        location_mse["metric"] = location_mse["squared_error"] ** 0.5

        # Return only the required columns
        return location_mse[["location", "metric"]]


class RMSEAggregate(MetricBase):
    """
    Fully aggregated Root Mean Squared Error metric.
    Computes a single RMSE value across all locations, time periods, and horizons.
    Aggregates directly from all data points, not by averaging per-location RMSEs.
    """

    spec = MetricSpec(
        output_dimensions=(),
        metric_name="RMSE",
        metric_id="rmse_aggregate",
        description="Aggregate RMSE across all data",
    )

    def compute(self, observations: FlatObserved, forecasts: FlatForecasts) -> pd.DataFrame:
        # Compute median forecast across samples for each location/time_period/horizon combination
        median_forecasts = forecasts.groupby(["location", "time_period", "horizon_distance"], as_index=False)[
            "forecast"
        ].median()

        # Merge observations with median forecasts on location and time_period
        merged = median_forecasts.merge(
            observations[["location", "time_period", "disease_cases"]], on=["location", "time_period"], how="inner"
        )

        # Calculate squared error from median prediction
        merged["squared_error"] = (merged["forecast"] - merged["disease_cases"]) ** 2

        # Average squared error across all entries, then take square root
        mse = merged["squared_error"].mean()
        rmse = mse**0.5

        return pd.DataFrame({"metric": [rmse]})


class DetailedRMSE(MetricBase):
    """
    Detailed Root Mean Squared Error metric.
    Does not group - gives one RMSE value per location/time_period/horizon_distance combination.
    This provides the highest resolution view of model performance.
    """

    spec = MetricSpec(
        output_dimensions=(DataDimension.location, DataDimension.time_period, DataDimension.horizon_distance),
        metric_name="RMSE",
        description="Detailed RMSE",
    )

    def compute(self, observations: pd.DataFrame, forecasts: pd.DataFrame) -> pd.DataFrame:
        # Compute median forecast across samples for each location/time_period/horizon combination
        median_forecasts = forecasts.groupby(["location", "time_period", "horizon_distance"], as_index=False)[
            "forecast"
        ].median()

        # Merge observations with median forecasts on location and time_period
        merged = median_forecasts.merge(
            observations[["location", "time_period", "disease_cases"]], on=["location", "time_period"], how="inner"
        )

        # Calculate squared error from median prediction
        merged["squared_error"] = (merged["forecast"] - merged["disease_cases"]) ** 2

        # Return RMSE per location/time_period/horizon combination
        merged["metric"] = merged["squared_error"] ** 0.5

        return merged[["location", "time_period", "horizon_distance", "metric"]]
