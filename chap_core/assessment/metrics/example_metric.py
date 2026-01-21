"""
Example metric for demonstration purposes.
"""

from chap_core.assessment.metrics.base import (
    AggregationOp,
    DeterministicMetric,
    MetricSpec,
)
from chap_core.assessment.metrics import metric


@metric()
class ExampleMetric(DeterministicMetric):
    """
    Example metric that computes absolute error.

    This is a demonstration metric showing how to create custom metrics
    using the metric system.

    Usage:
        example = ExampleMetric()
        detailed = example.get_detailed_metric(obs, forecasts)
        global_val = example.get_global_metric(obs, forecasts)
    """

    spec = MetricSpec(
        metric_id="example_metric",
        metric_name="Example Absolute Error",
        aggregation_op=AggregationOp.SUM,
        description="Sum of absolute error - demonstration metric",
    )

    def compute_point_metric(self, forecast: float, observed: float) -> float:
        """Compute absolute error for a single forecast/observation pair."""
        return abs(forecast - observed)
