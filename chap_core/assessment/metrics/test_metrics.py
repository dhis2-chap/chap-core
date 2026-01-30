"""
Test metrics for debugging and verification.
"""

import numpy as np
from chap_core.assessment.metrics.base import (
    AggregationOp,
    ProbabilisticMetric,
    MetricSpec,
)
from chap_core.assessment.metrics import metric


@metric()
class SampleCountMetric(ProbabilisticMetric):
    """
    Test metric that counts the number of forecast samples.

    Useful for debugging and verifying data structure correctness.
    Returns the count of samples at detailed level, sum when aggregated.

    Usage:
        sample_count = SampleCountMetric()
        detailed = sample_count.get_detailed_metric(obs, forecasts)
        global_val = sample_count.get_global_metric(obs, forecasts)
    """

    spec = MetricSpec(
        metric_id="sample_count",
        metric_name="Sample Count",
        aggregation_op=AggregationOp.SUM,
        description="Number of forecast samples (sum across aggregation)",
    )

    def compute_sample_metric(self, samples: np.ndarray, observed: float) -> float:
        """Return the count of samples."""
        return float(len(samples))
