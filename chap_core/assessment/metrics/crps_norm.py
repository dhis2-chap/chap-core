"""
Normalized Continuous Ranked Probability Score (CRPS) metric.
"""

import numpy as np

from chap_core.assessment.metrics import metric
from chap_core.assessment.metrics.base import (
    AggregationOp,
    MetricSpec,
    OptimizationDirection,
    ProbabilisticMetric,
)
from chap_core.assessment.metrics.crps import crps_rows


@metric()
class CRPSNormMetric(ProbabilisticMetric):
    """
    Normalized Continuous Ranked Probability Score (CRPS) metric.

    CRPS is normalized by the range of the observed values of the forecast cells
    (observations without a forecast are not used), to make it comparable across
    different scales.

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
        description="Normalized CRPS - CRPS divided by the range of the observed values that have a forecast",
        optimization_direction=OptimizationDirection.MINIMIZE,
        proper_scoring_rule=True,
        comparison_op="skill_ratio",
    )

    def compute_cell_metrics(self, samples: np.ndarray, observed: np.ndarray) -> np.ndarray:
        """CRPS of each cell divided by the range of the observations of the scored cells.

        Only observations that have a forecast count, so the score does not depend on how
        much of the dataset outside the evaluated periods is passed in.
        """
        crps = crps_rows(observed, samples)
        obs_range = observed.max() - observed.min()
        return crps / obs_range if obs_range > 0 else crps

    def compute_sample_metric(self, samples: np.ndarray, observed: float) -> float:
        """Not used directly - compute_detailed is overridden."""
        raise NotImplementedError("CRPSNormMetric overrides compute_cell_metrics")
