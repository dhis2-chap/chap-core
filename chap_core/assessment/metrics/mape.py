"""
Mean Absolute Percentage Error (MAPE) metric.
"""

import numpy as np

from chap_core.assessment.metrics import metric
from chap_core.assessment.metrics.base import (
    AggregationOp,
    DeterministicMetric,
    MetricSpec,
)


@metric()
class MAPEMetric(DeterministicMetric):
    """
    Mean Absolute Percentage Error metric.

    Computes the average of the absolute error for each entry. When aggregated using
    MEAN, this produces the MAPE (mean of absolute percentage error).

    Usage:
        mape = MAPEMetric()
        detailed = mape.get_detailed_metric(obs, forecasts)
        global_val = mape.get_global_metric(obs, forecasts)
        per_loc = mape.get_metric(obs, forecasts, dimensions=(DataDimension.location,))
    """

    spec = MetricSpec(
        metric_id="mape",
        metric_name="MAPE",
        aggregation_op=AggregationOp.MEAN,
        description="Mean Absolute Percentage Error - measures average absolute prediction error",
    )

    def compute_point_metric(self, forecast: float, observed: float) -> float:
        """Compute absolute error for a single forecast/observation pair."""
        if observed == 0.0:
            return np.nan
        return abs(observed - forecast) / abs(observed) * 100.0
