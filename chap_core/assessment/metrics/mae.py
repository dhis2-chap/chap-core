"""
Mean Absolute Error (MAE) metric.
"""

import pandas as pd
from chap_core.assessment.flat_representations import DataDimension
from chap_core.assessment.metrics.base import MetricBase, MetricSpec


class MAE(MetricBase):
    """
    Mean Absolute Error metric.
    Groups by location and horizon_distance to show error patterns across forecast horizons.
    """

    spec = MetricSpec(output_dimensions=(DataDimension.location, DataDimension.horizon_distance), metric_name="MAE")

    def compute(self, observations: pd.DataFrame, forecasts: pd.DataFrame) -> pd.DataFrame:
        # Merge observations with forecasts
        merged = forecasts.merge(
            observations[["location", "time_period", "disease_cases"]], 
            on=["location", "time_period"], 
            how="inner"
        )

        median_forecast = merged.groupby(
            ["location", "time_period", "horizon_distance"], as_index=False)["forecast"].median()
        
        median_with_truth = median_forecast.merge(
            observations[["location", "time_period", "disease_cases"]],
            on=["location", "time_period"],
            how="inner"
        )

        # Calculate absolute error
        median_with_truth["abs_error"] = (median_with_truth["forecast"] - median_with_truth["disease_cases"]).abs()

        # Then average across samples to get MAE per location and horizon
        mae_by_horizon = (
            median_with_truth.groupby(["location", "horizon_distance"], as_index=False)["abs_error"]
            .mean()
            .rename(columns={"abs_error": "metric"})
        )

        return mae_by_horizon
