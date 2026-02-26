"""
Winkler score metrics for evaluating prediction intervals.

The Winkler score (Winkler, 1972) rewards narrow prediction intervals but
penalizes when the observation falls outside the interval. For an interval
[L, U] at confidence level (1 - alpha):

- If L <= y <= U:  score = (U - L)
- If y < L:        score = (U - L) + (2 / alpha) * (L - y)
- If y > U:        score = (U - L) + (2 / alpha) * (y - U)

Lower scores are better.
"""

import numpy as np

from chap_core.assessment.metrics import metric
from chap_core.assessment.metrics.base import (
    AggregationOp,
    MetricSpec,
    ProbabilisticMetric,
)


class WinklerScoreMetric(ProbabilisticMetric):
    """
    Base class for Winkler score metrics.

    Computes the Winkler score for a prediction interval derived from
    forecast samples at given percentiles. Lower scores are better.

    This base class is not registered directly. Use concrete subclasses
    like WinklerScore10_90Metric or WinklerScore25_75Metric instead.
    """

    low_percentile: int
    high_percentile: int

    def compute_sample_metric(self, samples: np.ndarray, observed: float) -> float:
        """Compute Winkler score from samples and observation."""
        low, high = np.percentile(samples, [self.low_percentile, self.high_percentile])
        alpha = 1.0 - (self.high_percentile - self.low_percentile) / 100.0
        interval_width = high - low

        if observed < low:
            return float(interval_width + (2.0 / alpha) * (low - observed))
        elif observed > high:
            return float(interval_width + (2.0 / alpha) * (observed - high))
        else:
            return float(interval_width)


@metric()
class WinklerScore10_90Metric(WinklerScoreMetric):
    """Winkler score for 10th-90th percentile prediction interval."""

    spec = MetricSpec(
        metric_id="winkler_score_10_90",
        metric_name="Winkler Score 10-90",
        aggregation_op=AggregationOp.MEAN,
        description="Winkler score for 10th-90th percentile prediction interval",
    )
    low_percentile = 10
    high_percentile = 90


@metric()
class WinklerScore25_75Metric(WinklerScoreMetric):
    """Winkler score for 25th-75th percentile prediction interval."""

    spec = MetricSpec(
        metric_id="winkler_score_25_75",
        metric_name="Winkler Score 25-75",
        aggregation_op=AggregationOp.MEAN,
        description="Winkler score for 25th-75th percentile prediction interval",
    )
    low_percentile = 25
    high_percentile = 75
