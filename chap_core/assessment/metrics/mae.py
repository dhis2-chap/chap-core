"""
Mean Absolute Error (MAE) metric.
"""

from chap_core.assessment.metrics import metric
from chap_core.assessment.metrics.base import (
    AggregationOp,
    DeterministicMetric,
    MetricSpec,
)


@metric()
class MAEMetric(DeterministicMetric):
    """
    Mean Absolute Error metric.

    Computes absolute error at the detailed level. When aggregated using
    MEAN, this produces the MAE (mean of absolute errors).

    Usage:
        mae = MAEMetric()
        detailed = mae.get_detailed_metric(obs, forecasts)
        global_val = mae.get_global_metric(obs, forecasts)
        per_loc = mae.get_metric(obs, forecasts, dimensions=(DataDimension.location,))
    """

    spec = MetricSpec(
        metric_id="mae",
        metric_name="MAE",
        aggregation_op=AggregationOp.MEAN,
        description="Mean Absolute Error - measures average absolute prediction error",
    )

    def compute_point_metric(self, forecast: float, observed: float) -> float:
        """Compute absolute error for a single forecast/observation pair."""
        return abs(forecast - observed)
