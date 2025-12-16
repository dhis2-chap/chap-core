"""
Root Mean Squared Error (RMSE) metrics.
"""

import pandas as pd
from chap_core.assessment.flat_representations import DataDimension
from chap_core.assessment.metrics.base import DeterministicMetric, MetricSpec


class RMSE(DeterministicMetric):
    """
    Root Mean Squared Error metric.
    Groups by location to give RMSE per location across all time periods and horizons.
    """

    spec = MetricSpec(output_dimensions=(DataDimension.location,), metric_name="RMSE")

    def compute_from_merged(self, merged: pd.DataFrame) -> pd.DataFrame:
        merged["squared_error"] = (merged["forecast"] - merged["disease_cases"]) ** 2
        location_mse = merged.groupby("location", as_index=False)["squared_error"].mean()
        location_mse["metric"] = location_mse["squared_error"] ** 0.5
        return location_mse[["location", "metric"]]


class RMSEAggregate(DeterministicMetric):
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

    def compute_from_merged(self, merged: pd.DataFrame) -> pd.DataFrame:
        merged["squared_error"] = (merged["forecast"] - merged["disease_cases"]) ** 2
        mse = merged["squared_error"].mean()
        return pd.DataFrame({"metric": [mse**0.5]})


class DetailedRMSE(DeterministicMetric):
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

    def compute_from_merged(self, merged: pd.DataFrame) -> pd.DataFrame:
        merged["metric"] = ((merged["forecast"] - merged["disease_cases"]) ** 2) ** 0.5
        return merged[["location", "time_period", "horizon_distance", "metric"]]
