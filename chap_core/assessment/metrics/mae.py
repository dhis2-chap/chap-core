"""
Mean Absolute Error (MAE) metric.
"""

import pandas as pd
from chap_core.assessment.flat_representations import DataDimension
from chap_core.assessment.metrics.base import DeterministicMetric, MetricSpec


class MAE(DeterministicMetric):
    """
    Mean Absolute Error metric.
    Groups by location and horizon_distance to show error patterns across forecast horizons.
    """

    spec = MetricSpec(output_dimensions=(DataDimension.location, DataDimension.horizon_distance), metric_name="MAE")

    def compute_from_merged(self, merged: pd.DataFrame) -> pd.DataFrame:
        merged["abs_error"] = (merged["forecast"] - merged["disease_cases"]).abs()
        mae_by_horizon = (
            merged.groupby(["location", "horizon_distance"], as_index=False)["abs_error"]
            .mean()
            .rename(columns={"abs_error": "metric"})
        )
        return mae_by_horizon


class MAEAggregate(DeterministicMetric):
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

    def compute_from_merged(self, merged: pd.DataFrame) -> pd.DataFrame:
        merged["abs_error"] = (merged["forecast"] - merged["disease_cases"]).abs()
        return pd.DataFrame({"metric": [merged["abs_error"].mean()]})
