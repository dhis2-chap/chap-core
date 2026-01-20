"""
Mean Absolute Error (MAE) metric.
"""

from chap_core.assessment.metrics.base import (
    AggregationOp,
    DeterministicUnifiedMetric,
    UnifiedMetricSpec,
)


class MAEMetric(DeterministicUnifiedMetric):
    """
    Mean Absolute Error metric.

    Computes absolute error at the detailed level. When aggregated using
    MEAN, this produces the MAE (mean of absolute errors).

    Usage:
        mae = MAEMetric()
        detailed = mae.get_metric(obs, forecasts, AggregationLevel.DETAILED)
        per_loc = mae.get_metric(obs, forecasts, AggregationLevel.PER_LOCATION)
        aggregate = mae.get_metric(obs, forecasts, AggregationLevel.AGGREGATE)
    """

    spec = UnifiedMetricSpec(
        metric_id="mae",
        metric_name="MAE",
        aggregation_op=AggregationOp.MEAN,
        description="Mean Absolute Error - measures average absolute prediction error",
    )

    def compute_point_metric(self, forecast: float, observed: float) -> float:
        """Compute absolute error for a single forecast/observation pair."""
        return abs(forecast - observed)
