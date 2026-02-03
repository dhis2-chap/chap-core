"""
Metrics for measuring forecast bias (samples above truth).
"""

import numpy as np

from chap_core.assessment.metrics.base import (
    AggregationOp,
    ProbabilisticMetric,
    MetricSpec,
)
from chap_core.assessment.metrics import metric


@metric()
class RatioAboveTruthMetric(ProbabilisticMetric):
    """
    Ratio of forecast samples above the observed truth.

    Computes the proportion of forecast samples that exceed the observed value.
    A value of 0.5 indicates unbiased forecasts, values > 0.5 indicate
    over-prediction bias, and values < 0.5 indicate under-prediction bias.

    Usage:
        ratio_above = RatioAboveTruthMetric()
        detailed = ratio_above.get_detailed_metric(obs, forecasts)
        global_val = ratio_above.get_global_metric(obs, forecasts)
        per_loc = ratio_above.get_metric(obs, forecasts, dimensions=(DataDimension.location,))
    """

    spec = MetricSpec(
        metric_id="ratio_above_truth",
        metric_name="Ratio Above Truth",
        aggregation_op=AggregationOp.MEAN,
        description="Proportion of forecast samples exceeding the observed value (0.5 = unbiased)",
    )

    def compute_sample_metric(self, samples: np.ndarray, observed: float) -> float:
        """Compute ratio of samples above the observation."""
        above_count: int = int(np.sum(samples > observed))
        return float(above_count / len(samples))
