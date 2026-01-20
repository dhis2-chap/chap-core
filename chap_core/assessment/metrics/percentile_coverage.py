"""
Percentile coverage metrics for evaluating forecast calibration.
"""

import numpy as np

from chap_core.assessment.metrics.base import (
    AggregationOp,
    ProbabilisticUnifiedMetric,
    UnifiedMetricSpec,
)


class PercentileCoverageMetric(ProbabilisticUnifiedMetric):
    """
    Percentile coverage metric.

    Computes whether the observation falls within the specified percentile range
    of the forecast samples. Returns 1 if within range, 0 otherwise at the
    detailed level. When aggregated, this gives the proportion of observations
    within the range.

    Usage:
        # 10th-90th percentile coverage
        coverage_10_90 = PercentileCoverageMetric(10, 90)
        result = coverage_10_90.get_metric(obs, forecasts, AggregationLevel.AGGREGATE)

        # 25th-75th percentile coverage
        coverage_25_75 = PercentileCoverageMetric(25, 75)
        result = coverage_25_75.get_metric(obs, forecasts, AggregationLevel.AGGREGATE)
    """

    def __init__(self, low_percentile: int = 10, high_percentile: int = 90):
        """
        Initialize the percentile coverage metric.

        Args:
            low_percentile: Lower percentile bound (e.g., 10 for 10th percentile)
            high_percentile: Upper percentile bound (e.g., 90 for 90th percentile)
        """
        self.low_percentile = low_percentile
        self.high_percentile = high_percentile
        self.spec = UnifiedMetricSpec(
            metric_id=f"coverage_{low_percentile}_{high_percentile}",
            metric_name=f"Coverage {low_percentile}-{high_percentile}",
            aggregation_op=AggregationOp.MEAN,
            description=f"Proportion of observations within {low_percentile}th-{high_percentile}th percentile",
        )

    def compute_sample_metric(self, samples: np.ndarray, observed: float) -> float:
        """Check if observation is within the percentile range."""
        low, high = np.percentile(samples, [self.low_percentile, self.high_percentile])
        return 1.0 if (low <= observed <= high) else 0.0


# Factory functions for common coverage metrics
def Coverage10_90Metric() -> PercentileCoverageMetric:
    """Create a 10th-90th percentile coverage metric."""
    return PercentileCoverageMetric(10, 90)


def Coverage25_75Metric() -> PercentileCoverageMetric:
    """Create a 25th-75th percentile coverage metric."""
    return PercentileCoverageMetric(25, 75)
