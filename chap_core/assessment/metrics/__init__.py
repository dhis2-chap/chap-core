"""
Metrics submodule for assessment.

This module provides a metric system where each metric is defined once
and supports multiple aggregation levels via dimension specification.

Metrics are registered using the @metric() decorator, which automatically
adds them to the registry based on their spec.metric_id.
"""

import logging
from typing import Callable

from chap_core.assessment.evaluation import Evaluation
from chap_core.assessment.flat_representations import FlatForecasts, FlatObserved, DataDimension
from chap_core.database.tables import BackTest
from chap_core.assessment.metrics.base import (
    AggregationOp,
    Metric,
    MetricSpec,
    DeterministicMetric,
    ProbabilisticMetric,
    DEFAULT_OUTPUT_DIMENSIONS,
)

logger = logging.getLogger(__name__)

# Registry mapping metric_id to metric class
_metrics_registry: dict[str, type[Metric]] = {}


def metric():
    """Decorator to register a metric class. Reads metric_id from class spec."""

    def decorator(cls: type[Metric]) -> type[Metric]:
        if not isinstance(cls, type) or not issubclass(cls, Metric):
            raise TypeError(f"{cls} must be a class inheriting from Metric")
        if not hasattr(cls, "spec"):
            raise ValueError(f"{cls.__name__} missing 'spec' class attribute")

        _metrics_registry[cls.spec.metric_id] = cls
        return cls

    return decorator


def get_metrics_registry() -> dict[str, type[Metric]]:
    """Get a copy of the metrics registry."""
    return _metrics_registry.copy()


def get_metric(metric_id: str) -> type[Metric] | None:
    """Get a metric class by ID."""
    return _metrics_registry.get(metric_id)


def list_metrics() -> list[dict]:
    """List all registered metrics with metadata (id, name, description, aggregation_op)."""
    result = []
    for metric_id, metric_cls in _metrics_registry.items():
        spec = metric_cls.spec
        result.append(
            {
                "id": spec.metric_id,
                "name": spec.metric_name,
                "description": spec.description,
                "aggregation_op": spec.aggregation_op.value,
            }
        )
    return result


def _discover_metrics():
    """Import all metric modules to trigger registration."""
    # Import each module to trigger @metric() decorators
    from chap_core.assessment.metrics import rmse  # noqa: F401
    from chap_core.assessment.metrics import mae  # noqa: F401
    from chap_core.assessment.metrics import crps  # noqa: F401
    from chap_core.assessment.metrics import crps_norm  # noqa: F401
    from chap_core.assessment.metrics import above_truth  # noqa: F401
    from chap_core.assessment.metrics import percentile_coverage  # noqa: F401
    from chap_core.assessment.metrics import test_metrics  # noqa: F401
    from chap_core.assessment.metrics import example_metric  # noqa: F401


# Discover metrics at module load
_discover_metrics()

# Import metric classes for backwards compatibility in exports
from chap_core.assessment.metrics.rmse import RMSEMetric  # noqa: E402
from chap_core.assessment.metrics.mae import MAEMetric  # noqa: E402
from chap_core.assessment.metrics.crps import CRPSMetric  # noqa: E402
from chap_core.assessment.metrics.crps_norm import CRPSNormMetric  # noqa: E402
from chap_core.assessment.metrics.peak_diff import PeakValueDiffMetric, PeakPeriodLagMetric  # noqa: E402
from chap_core.assessment.metrics.above_truth import RatioAboveTruthMetric  # noqa: E402
from chap_core.assessment.metrics.percentile_coverage import (  # noqa: E402
    PercentileCoverageMetric,
    Coverage10_90Metric,
    Coverage25_75Metric,
)
from chap_core.assessment.metrics.test_metrics import SampleCountMetric  # noqa: E402
from chap_core.assessment.metrics.example_metric import ExampleMetric  # noqa: E402

# Backward compatibility alias
available_metrics: dict[str, Callable[[], Metric]] = _metrics_registry

__all__ = [
    # Base classes
    "AggregationOp",
    "Metric",
    "MetricSpec",
    "DeterministicMetric",
    "ProbabilisticMetric",
    "DEFAULT_OUTPUT_DIMENSIONS",
    "DataDimension",
    # Registry API
    "metric",
    "get_metrics_registry",
    "get_metric",
    "list_metrics",
    "available_metrics",
    # Metrics
    "RMSEMetric",
    "MAEMetric",
    "CRPSMetric",
    "CRPSNormMetric",
    "PeakValueDiffMetric",
    "PeakPeriodLagMetric",
    "RatioAboveTruthMetric",
    "PercentileCoverageMetric",
    "Coverage10_90Metric",
    "Coverage25_75Metric",
    "SampleCountMetric",
    "ExampleMetric",
]


def compute_all_aggregated_metrics_from_backtest(backtest: BackTest) -> dict[str, float]:
    """
    Compute all available metrics as global aggregated values for a backtest.

    Args:
        backtest: The BackTest object to compute metrics for

    Returns:
        Dictionary mapping metric_id to the aggregated metric value
    """
    logger.info(f"Computing aggregated metrics for backtest {backtest.id}")

    # Use Evaluation abstraction to get flat representation
    evaluation = Evaluation.from_backtest(backtest)
    flat_data = evaluation.to_flat()

    results = {}
    for metric_id, metric_factory in available_metrics.items():
        metric = metric_factory()
        metric_df = metric.get_global_metric(flat_data.observations, flat_data.forecasts)
        if len(metric_df) != 1:
            raise ValueError(
                f"Metric {metric_id} was expected to return a single aggregated value, but got {len(metric_df)} rows."
            )
        results[metric_id] = float(metric_df["metric"].iloc[0])

    logger.info(f"Computed metrics: {list(results.keys())}")
    return results
