"""
Normalized Continuous Ranked Probability Score (CRPS) metric.
"""

import numpy as np
import pandas as pd

from chap_core.assessment.metrics import metric
from chap_core.assessment.metrics.base import (
    AggregationOp,
    MetricSpec,
    ProbabilisticMetric,
)
from chap_core.assessment.metrics.crps import CRPSMetric


@metric()
class CRPSNormMetric(ProbabilisticMetric):
    """
    Normalized Continuous Ranked Probability Score (CRPS) metric.

    CRPS is normalized by the range of observed values to make it comparable
    across different scales.

    Usage:
        crps_norm = CRPSNormMetric()
        detailed = crps_norm.get_detailed_metric(obs, forecasts)
        global_val = crps_norm.get_global_metric(obs, forecasts)
        per_loc = crps_norm.get_metric(obs, forecasts, dimensions=(DataDimension.location,))
    """

    spec = MetricSpec(
        metric_id="crps_norm",
        metric_name="CRPS Normalized",
        aggregation_op=AggregationOp.MEAN,
        description="Normalized CRPS - CRPS divided by the range of observed values",
    )

    def compute_detailed(self, observations: pd.DataFrame, forecasts: pd.DataFrame) -> pd.DataFrame:
        """Compute normalized CRPS using all samples."""
        # First compute regular CRPS
        crps_metric = CRPSMetric()
        detailed_crps = crps_metric.compute_detailed(observations, forecasts)

        # Calculate normalization factor based on range of all observed values
        obs_values = observations["disease_cases"].values
        obs_min, obs_max = obs_values.min(), obs_values.max()
        obs_range = obs_max - obs_min  # type: ignore[operator]

        # Normalize CRPS by the range (avoid division by zero)
        if obs_range > 0:  # type: ignore[operator]
            detailed_crps["metric"] = detailed_crps["metric"] / obs_range  # type: ignore[operator]

        return detailed_crps

    def compute_sample_metric(self, samples: np.ndarray, observed: float) -> float:
        """Not used directly - compute_detailed is overridden."""
        raise NotImplementedError("CRPSNormMetric overrides compute_detailed")
