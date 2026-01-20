"""
Root Mean Squared Error (RMSE) metric.
"""

from chap_core.assessment.metrics.base import (
    AggregationOp,
    DeterministicUnifiedMetric,
    UnifiedMetricSpec,
)


class RMSEMetric(DeterministicUnifiedMetric):
    """
    Root Mean Squared Error metric.

    Computes absolute error at the detailed level. When aggregated using
    ROOT_MEAN_SQUARE, this produces the RMSE (sqrt of mean squared error).

    Usage:
        rmse = RMSEMetric()
        detailed = rmse.get_metric(obs, forecasts, AggregationLevel.DETAILED)
        per_loc = rmse.get_metric(obs, forecasts, AggregationLevel.PER_LOCATION)
        aggregate = rmse.get_metric(obs, forecasts, AggregationLevel.AGGREGATE)
    """

    spec = UnifiedMetricSpec(
        metric_id="rmse",
        metric_name="RMSE",
        aggregation_op=AggregationOp.ROOT_MEAN_SQUARE,
        description="Root Mean Squared Error - measures average prediction error magnitude",
    )

    def compute_point_metric(self, forecast: float, observed: float) -> float:
        """Compute absolute error for a single forecast/observation pair."""
        return abs(forecast - observed)
