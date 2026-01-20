"""
Normalized Continuous Ranked Probability Score (CRPS) metric.
"""

import numpy as np
import pandas as pd

from chap_core.assessment.metrics.base import (
    AggregationOp,
    ProbabilisticUnifiedMetric,
    UnifiedMetricSpec,
)
from chap_core.assessment.metrics.crps import CRPSMetric


class CRPSNormMetric(ProbabilisticUnifiedMetric):
    """
    Normalized Continuous Ranked Probability Score (CRPS) metric.

    CRPS is normalized by the range of observed values to make it comparable
    across different scales.

    Usage:
        crps_norm = CRPSNormMetric()
        detailed = crps_norm.get_metric(obs, forecasts, AggregationLevel.DETAILED)
        per_loc = crps_norm.get_metric(obs, forecasts, AggregationLevel.PER_LOCATION)
        aggregate = crps_norm.get_metric(obs, forecasts, AggregationLevel.AGGREGATE)
    """

    spec = UnifiedMetricSpec(
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
        obs_range = obs_max - obs_min

        # Normalize CRPS by the range (avoid division by zero)
        if obs_range > 0:
            detailed_crps["metric"] = detailed_crps["metric"] / obs_range

        return detailed_crps

    def compute_sample_metric(self, samples: np.ndarray, observed: float) -> float:
        """Not used directly - compute_detailed is overridden."""
        raise NotImplementedError("CRPSNormMetric overrides compute_detailed")
