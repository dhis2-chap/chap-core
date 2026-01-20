"""
Metrics for measuring forecast bias (samples above truth).
"""

import numpy as np

from chap_core.assessment.metrics.base import (
    AggregationOp,
    ProbabilisticUnifiedMetric,
    UnifiedMetricSpec,
)


class RatioAboveTruthMetric(ProbabilisticUnifiedMetric):
    """
    Ratio of forecast samples above the observed truth.

    Computes the proportion of forecast samples that exceed the observed value.
    A value of 0.5 indicates unbiased forecasts, values > 0.5 indicate
    over-prediction bias, and values < 0.5 indicate under-prediction bias.

    Usage:
        ratio_above = RatioAboveTruthMetric()
        detailed = ratio_above.get_metric(obs, forecasts, AggregationLevel.DETAILED)
        per_loc = ratio_above.get_metric(obs, forecasts, AggregationLevel.PER_LOCATION)
        aggregate = ratio_above.get_metric(obs, forecasts, AggregationLevel.AGGREGATE)
    """

    spec = UnifiedMetricSpec(
        metric_id="ratio_above_truth",
        metric_name="Ratio Above Truth",
        aggregation_op=AggregationOp.MEAN,
        description="Proportion of forecast samples exceeding the observed value (0.5 = unbiased)",
    )

    def compute_sample_metric(self, samples: np.ndarray, observed: float) -> float:
        """Compute ratio of samples above the observation."""
        above_count = np.sum(samples > observed)
        return float(above_count / len(samples))
