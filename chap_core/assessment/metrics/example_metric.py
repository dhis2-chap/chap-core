"""
Example metric for demonstration purposes.
"""

import pandas as pd
from chap_core.assessment.flat_representations import DataDimension, FlatForecasts, FlatObserved
from chap_core.assessment.metrics.base import MetricBase, MetricSpec


class ExampleMetric(MetricBase):
    """
    Example metric that computes absolute error per location and time_period.
    This is a demonstration metric showing how to create custom metrics.
    """

    spec = MetricSpec(
        output_dimensions=(DataDimension.location, DataDimension.time_period),
        metric_name="Example Absolute Error",
        metric_id="example_metric",
        description="Sum of absolute error per location and time_period",
    )

    def compute(self, observations: FlatObserved, forecasts: FlatForecasts) -> pd.DataFrame:
        # sum of absolute error per location and time_period
        merged = forecasts.merge(observations, on=["location", "time_period"], how="left")
        merged["metric"] = (merged["forecast"] - merged["disease_cases"]).abs()
        return merged[["location", "time_period", "metric"]]
