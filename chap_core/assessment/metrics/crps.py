"""
Continuous Ranked Probability Score (CRPS) metric.
"""

import numpy as np

from chap_core.assessment.metrics import metric
from chap_core.assessment.metrics.base import (
    AggregationOp,
    MetricSpec,
    ProbabilisticMetric,
)


def _crps_sample(samples: np.ndarray, observed: float) -> float:
    """CRPS estimate from forecast samples via the order-statistic formula.

    Equivalent to ``E[|X - obs|] - 0.5 * E[|X - X'|]`` but computed as
    ``E[|X - obs|] - (1/n^2) * sum((2i - n - 1) * x_(i))`` with ``x_(i)`` the
    sorted samples. This is one sort plus one reduction (O(n log n) time, O(n)
    memory) instead of the O(n^2) pairwise difference matrix, giving identical
    results ~25x faster on large sample counts.
    """
    n = samples.shape[0]
    term1 = float(np.mean(np.abs(samples - observed)))
    if n == 0:
        return term1
    xs = np.sort(samples)
    coeffs = 2.0 * np.arange(1, n + 1) - n - 1
    term2 = float(np.sum(coeffs * xs) / (n * n))
    return term1 - term2


@metric()
class CRPSMetric(ProbabilisticMetric):
    """
    Continuous Ranked Probability Score (CRPS) metric.

    CRPS measures both calibration and sharpness of probabilistic forecasts.
    It is computed using all forecast samples.

    Formula: CRPS = E[|X - obs|] - 0.5 * E[|X - X'|]
    where X and X' are independent samples from the forecast distribution.

    Usage:
        crps = CRPSMetric()
        detailed = crps.get_detailed_metric(obs, forecasts)
        global_val = crps.get_global_metric(obs, forecasts)
        per_loc = crps.get_metric(obs, forecasts, dimensions=(DataDimension.location,))
    """

    spec = MetricSpec(
        metric_id="crps",
        metric_name="CRPS",
        aggregation_op=AggregationOp.MEAN,
        description="Continuous Ranked Probability Score - measures calibration and sharpness",
    )

    def compute_sample_metric(self, samples: np.ndarray, observed: float) -> float:
        """Compute CRPS from all samples and the observation."""
        return _crps_sample(samples, observed)


@metric()
class CRPSLog1pMetric(ProbabilisticMetric):
    """
    CRPS on log1p-transformed values.

    Applies log1p to both forecast samples and observations before computing
    CRPS. This reduces the influence of large values and makes the metric
    more sensitive to relative errors.
    """

    spec = MetricSpec(
        metric_id="crps_log1p",
        metric_name="CRPS (log1p)",
        aggregation_op=AggregationOp.MEAN,
        description="CRPS computed on log(1+x)-transformed forecasts and observations",
    )

    def compute_sample_metric(self, samples: np.ndarray, observed: float) -> float:
        """Compute CRPS on log1p-transformed values."""
        return _crps_sample(np.log1p(samples), float(np.log1p(observed)))
