"""
Root Mean Squared Error (RMSE) metric.
"""

from chap_core.assessment.metrics import metric
from chap_core.assessment.metrics.base import (
    AggregationOp,
    DeterministicMetric,
    MetricSpec,
)


@metric()
class RMSEMetric(DeterministicMetric):
    """
    Root Mean Squared Error metric.

    Computes absolute error at the detailed level. When aggregated using
    ROOT_MEAN_SQUARE, this produces the RMSE (sqrt of mean squared error).

    Usage:
        rmse = RMSEMetric()
        detailed = rmse.get_detailed_metric(obs, forecasts)
        global_val = rmse.get_global_metric(obs, forecasts)
        per_loc = rmse.get_metric(obs, forecasts, dimensions=(DataDimension.location,))
    """

    spec = MetricSpec(
        metric_id="rmse",
        metric_name="RMSE",
        aggregation_op=AggregationOp.ROOT_MEAN_SQUARE,
        description="Root Mean Squared Error - measures average prediction error magnitude",
    )

    def compute_point_metric(self, forecast: float, observed: float) -> float:
        """Compute absolute error for a single forecast/observation pair."""
        return abs(forecast - observed)
