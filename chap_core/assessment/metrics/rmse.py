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
        # Merge observations with forecasts on location and time_period
        merged = forecasts.merge(
            observations[["location", "time_period", "disease_cases"]], on=["location", "time_period"], how="inner"
        )

        # Calculate squared error for each forecast
        merged["squared_error"] = (merged["forecast"] - merged["disease_cases"]) ** 2

        # First average across samples for each location/time_period combination
        per_sample_mse = merged.groupby(["location", "time_period", "sample"], as_index=False)["squared_error"].mean()

        # Then average across all time periods and samples for each location
        location_mse = per_sample_mse.groupby("location", as_index=False)["squared_error"].mean()

        # Take square root to get RMSE
        location_mse["metric"] = location_mse["squared_error"] ** 0.5

        # Return only the required columns
        return location_mse[["location", "metric"]]


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
        # Merge observations with forecasts on location and time_period
        merged = forecasts.merge(
            observations[["location", "time_period", "disease_cases"]], on=["location", "time_period"], how="inner"
        )

        # Calculate squared error for each forecast
        merged["squared_error"] = (merged["forecast"] - merged["disease_cases"]) ** 2

        # Average across samples for each location/time_period/horizon combination
        detailed_mse = merged.groupby(["location", "time_period", "horizon_distance"], as_index=False)[
            "squared_error"
        ].mean()

        # Take square root to get RMSE
        detailed_mse["metric"] = detailed_mse["squared_error"] ** 0.5

        # Return only the required columns
        return detailed_mse[["location", "time_period", "horizon_distance", "metric"]]
