"""
Metrics submodule for assessment.

This module provides a unified metric system where each metric is defined once
and supports multiple aggregation levels (DETAILED, PER_LOCATION, PER_HORIZON, AGGREGATE).
"""

import logging
from typing import Callable

from chap_core.assessment.evaluation import Evaluation
from chap_core.assessment.flat_representations import FlatForecasts, FlatObserved
from chap_core.database.tables import BackTest
from chap_core.assessment.metrics.base import (
    AggregationLevel,
    AggregationOp,
    UnifiedMetric,
    UnifiedMetricSpec,
    DeterministicUnifiedMetric,
    ProbabilisticUnifiedMetric,
    LEVEL_TO_DIMENSIONS,
)
from chap_core.assessment.metrics.rmse import RMSEMetric
from chap_core.assessment.metrics.mae import MAEMetric
from chap_core.assessment.metrics.crps import CRPSMetric
from chap_core.assessment.metrics.crps_norm import CRPSNormMetric
from chap_core.assessment.metrics.peak_diff import PeakValueDiffMetric, PeakPeriodLagMetric
from chap_core.assessment.metrics.above_truth import RatioAboveTruthMetric
from chap_core.assessment.metrics.percentile_coverage import (
    PercentileCoverageMetric,
    Coverage10_90Metric,
    Coverage25_75Metric,
)
from chap_core.assessment.metrics.test_metrics import SampleCountMetric
from chap_core.assessment.metrics.example_metric import ExampleMetric

logger = logging.getLogger(__name__)

__all__ = [
    # Base classes
    "AggregationLevel",
    "AggregationOp",
    "UnifiedMetric",
    "UnifiedMetricSpec",
    "DeterministicUnifiedMetric",
    "ProbabilisticUnifiedMetric",
    "LEVEL_TO_DIMENSIONS",
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

# Dictionary of available metrics for easy lookup
# Each entry maps a metric_id to a callable that returns a metric instance
available_metrics: dict[str, Callable[[], UnifiedMetric]] = {
    "rmse": RMSEMetric,
    "mae": MAEMetric,
    "crps": CRPSMetric,
    "crps_norm": CRPSNormMetric,
    "ratio_above_truth": RatioAboveTruthMetric,
    "coverage_10_90": Coverage10_90Metric,
    "coverage_25_75": Coverage25_75Metric,
    "sample_count": SampleCountMetric,
    "example_metric": ExampleMetric,
    # Peak metrics are special - they compute across time periods
    # "peak_value_diff": PeakValueDiffMetric,
    # "peak_period_lag": PeakPeriodLagMetric,
}


def compute_all_aggregated_metrics_from_backtest(backtest: BackTest) -> dict[str, float]:
    """
    Compute all available metrics at the AGGREGATE level for a backtest.

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
        metric_df = metric.get_metric(flat_data.observations, flat_data.forecasts, AggregationLevel.AGGREGATE)
        if len(metric_df) != 1:
            raise ValueError(
                f"Metric {metric_id} was expected to return a single aggregated value, but got {len(metric_df)} rows."
            )
        results[metric_id] = float(metric_df["metric"].iloc[0])

    logger.info(f"Computed metrics: {list(results.keys())}")
    return results
