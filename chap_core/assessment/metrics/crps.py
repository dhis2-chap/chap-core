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


def crps_score_unbiased(samples: np.ndarray, observed: float) -> float:
    """Compute the unbiased O(m log m) CRPS for a single observation."""
    samples = np.asarray(samples, float).reshape(-1)
    term1 = np.mean(np.abs(samples - observed))
    m = samples.size
    if m <= 1:
        return float(term1)

    sorted_s = np.sort(samples)
    cumsum_s = np.cumsum(sorted_s)
    k = np.arange(m)

    left = sorted_s * k - cumsum_s + sorted_s
    rev_cumsum_s = np.cumsum(sorted_s[::-1])[::-1]
    right = rev_cumsum_s - sorted_s * (m - k)
    pairwise = left + right

    sum_pairwise = 0.5 * np.sum(pairwise)
    denom = m * (m - 1) / 2.0
    term2 = sum_pairwise / denom

    return float(term1 - 0.5 * term2)


def crps_score_unbiased_matrix(observations: np.ndarray, forecasts: np.ndarray) -> float:
    """Compute the unbiased CRPS over multiple observations and sample matrices."""
    observations = np.asarray(observations, float).reshape(-1)
    forecasts = np.asarray(forecasts, float)
    if forecasts.ndim != 2:
        raise ValueError(f"forecasts must be 2D (n, m), got shape {forecasts.shape}")
    n, m = forecasts.shape
    if n != observations.shape[0]:
        raise ValueError(f"observations length {observations.shape[0]} does not match forecast rows {n}")

    term1 = np.mean(np.abs(forecasts - observations[:, None]), axis=1)

    if m <= 1:
        return float(np.mean(term1))

    sorted_f = np.sort(forecasts, axis=1)
    cumsum_f = np.cumsum(sorted_f, axis=1)
    k = np.arange(m)

    left = sorted_f * k - cumsum_f + sorted_f

    rev_cumsum_f = np.cumsum(sorted_f[:, ::-1], axis=1)[:, ::-1]
    right = rev_cumsum_f - sorted_f * (m - k)

    pairwise = left + right
    sum_pairwise = 0.5 * np.sum(pairwise, axis=1)

    denom = m * (m - 1) / 2.0
    term2 = sum_pairwise / denom

    return float(np.mean(term1 - 0.5 * term2))


@metric()
class CRPSMetric(ProbabilisticMetric):
    """
    Continuous Ranked Probability Score (CRPS) metric.

    CRPS measures both calibration and sharpness of probabilistic forecasts
    and is computed using all forecast samples.

    Mathematically:
        CRPS(F, y) = E[|X - y|] - 0.5 * E[|X - X'|],
    where X and X' are independent draws from the forecast distribution F.

    This implementation uses the unbiased ("fair") estimator with factor
    1 / (m * (m - 1)) over m samples, and an O(m log m) algorithm based on
    sorting and cumulative sums, replacing the previous naive O(m^2)
    pairwise implementation.

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
        return crps_score_unbiased(samples, observed)


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
        log_samples = np.log1p(samples)
        log_observed = np.log1p(observed)
        return crps_score_unbiased(log_samples, log_observed)
