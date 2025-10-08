"""
Test metrics for debugging and verification.
"""

import pandas as pd
from chap_core.assessment.flat_representations import DataDimension, FlatForecasts, FlatObserved
from chap_core.assessment.metrics.base import MetricBase, MetricSpec


class TestMetricDetailed(MetricBase):
    """
    Test metric that counts the number of forecast samples per location/time_period/horizon_distance.
    Useful for debugging and verifying data structure correctness.
    Returns the count of samples for each combination.
    """

    spec = MetricSpec(
        output_dimensions=(DataDimension.location, DataDimension.time_period, DataDimension.horizon_distance),
        metric_name="Sample Count",
        metric_id="test_sample_count_detailed",
        description="Number of forecast samples per location, time period and horizon",
    )

    def compute(self, observations: FlatObserved, forecasts: FlatForecasts) -> pd.DataFrame:
        # Group by location, time_period, and horizon_distance to count samples
        sample_counts = (
            forecasts.groupby(["location", "time_period", "horizon_distance"], as_index=False)
            .size()
            .rename(columns={"size": "metric"})
        )

        # Convert metric to float to match schema expectations
        sample_counts["metric"] = sample_counts["metric"].astype(float)

        return sample_counts


class TestMetric(MetricBase):
    """
    Test metric that counts the total number of forecast samples in the entire dataset.
    Returns a single number representing the total sample count.
    """

    spec = MetricSpec(
        output_dimensions=(),
        metric_name="Sample Count",
        metric_id="test_sample_count",
        description="Total number of forecast samples in dataset",
    )

    def compute(self, observations: FlatObserved, forecasts: FlatForecasts) -> pd.DataFrame:
        # Count total number of rows in forecasts (each row is one sample)
        total_samples = float(len(forecasts))
        return pd.DataFrame({"metric": [total_samples]})
