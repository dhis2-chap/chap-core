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
        # CRPS = E[|X - obs|] - 0.5 * E[|X - X'|]
        term1 = np.mean(np.abs(samples - observed))
        term2 = 0.5 * np.mean(np.abs(samples[:, None] - samples[None, :]))
        return float(term1 - term2)


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
        term1 = np.mean(np.abs(log_samples - log_observed))
        term2 = 0.5 * np.mean(np.abs(log_samples[:, None] - log_samples[None, :]))
        return float(term1 - term2)
