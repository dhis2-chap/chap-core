"""
Percentile coverage metrics for evaluating forecast calibration.
"""

import numpy as np

from chap_core.assessment.metrics.base import (
    AggregationOp,
    ProbabilisticUnifiedMetric,
    UnifiedMetricSpec,
)
from chap_core.assessment.metrics import metric


class PercentileCoverageMetric(ProbabilisticUnifiedMetric):
    """
    Base class for percentile coverage metrics.

    Computes whether the observation falls within the specified percentile range
    of the forecast samples. Returns 1 if within range, 0 otherwise at the
    detailed level. When aggregated, this gives the proportion of observations
    within the range.

    This base class is not registered directly. Use concrete subclasses
    like Coverage10_90Metric or Coverage25_75Metric instead.
    """

    low_percentile: int
    high_percentile: int

    def compute_sample_metric(self, samples: np.ndarray, observed: float) -> float:
        """Check if observation is within the percentile range."""
        low, high = np.percentile(samples, [self.low_percentile, self.high_percentile])
        return 1.0 if (low <= observed <= high) else 0.0


@metric()
class Coverage10_90Metric(PercentileCoverageMetric):
    """10th-90th percentile coverage metric."""

    spec = UnifiedMetricSpec(
        metric_id="coverage_10_90",
        metric_name="Coverage 10-90",
        aggregation_op=AggregationOp.MEAN,
        description="Proportion of observations within 10th-90th percentile",
    )
    low_percentile = 10
    high_percentile = 90


@metric()
class Coverage25_75Metric(PercentileCoverageMetric):
    """25th-75th percentile coverage metric."""

    spec = UnifiedMetricSpec(
        metric_id="coverage_25_75",
        metric_name="Coverage 25-75",
        aggregation_op=AggregationOp.MEAN,
        description="Proportion of observations within 25th-75th percentile",
    )
    low_percentile = 25
    high_percentile = 75
