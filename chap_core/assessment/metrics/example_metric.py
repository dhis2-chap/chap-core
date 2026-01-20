"""
Example metric for demonstration purposes.
"""

from chap_core.assessment.metrics.base import (
    AggregationOp,
    DeterministicUnifiedMetric,
    UnifiedMetricSpec,
)


class ExampleMetric(DeterministicUnifiedMetric):
    """
    Example metric that computes absolute error.

    This is a demonstration metric showing how to create custom metrics
    using the unified metric system.

    Usage:
        example = ExampleMetric()
        detailed = example.get_metric(obs, forecasts, AggregationLevel.DETAILED)
        aggregate = example.get_metric(obs, forecasts, AggregationLevel.AGGREGATE)
    """

    spec = UnifiedMetricSpec(
        metric_id="example_metric",
        metric_name="Example Absolute Error",
        aggregation_op=AggregationOp.SUM,
        description="Sum of absolute error - demonstration metric",
    )

    def compute_point_metric(self, forecast: float, observed: float) -> float:
        """Compute absolute error for a single forecast/observation pair."""
        return abs(forecast - observed)
