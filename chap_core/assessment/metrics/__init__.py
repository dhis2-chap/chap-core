"""
Metrics submodule for assessment.
All metrics are imported here for backwards compatibility.
"""

import logging
from chap_core.assessment.evaluation import Evaluation
from chap_core.assessment.flat_representations import FlatForecasts, FlatObserved
from chap_core.database.tables import BackTest
from chap_core.assessment.metrics.base import MetricBase, MetricSpec
from chap_core.assessment.metrics.rmse import RMSE, RMSEAggregate, DetailedRMSE
from chap_core.assessment.metrics.mae import MAE, MAEAggregate
from chap_core.assessment.metrics.crps import CRPS, CRPSPerLocation, DetailedCRPS
from chap_core.assessment.metrics.crps_norm import CRPSNorm, DetailedCRPSNorm
from chap_core.assessment.metrics.peak_diff import PeakValueDiffMetric, PeakPeriodLagMetric
from chap_core.assessment.metrics.above_truth import SamplesAboveTruth
from chap_core.assessment.metrics.percentile_coverage import (
    IsWithin10th90thDetailed,
    IsWithin25th75thDetailed,
    RatioWithin10th90th,
    RatioWithin10th90thPerLocation,
    RatioWithin25th75th,
    RatioWithin25th75thPerLocation,
)
from chap_core.assessment.metrics.test_metrics import TestMetric, TestMetricDetailed
from chap_core.assessment.metrics.example_metric import ExampleMetric

logger = logging.getLogger(__name__)

__all__ = [
    "MetricBase",
    "MetricSpec",
    "RMSE",
    "RMSEAggregate",
    "DetailedRMSE",
    "MAE",
    "MAEAggregate",
    "CRPS",
    "CRPSPerLocation",
    "DetailedCRPS",
    "CRPSNorm",
    "DetailedCRPSNorm",
    "PeakValueDiffMetric",
    "PeakPeriodLagMetric",
    "SamplesAboveTruth",
    "IsWithin10th90thDetailed",
    "IsWithin25th75thDetailed",
    "RatioWithin10th90th",
    "RatioWithin10th90thPerLocation",
    "RatioWithin25th75th",
    "RatioWithin25th75thPerLocation",
    "TestMetric",
    "TestMetricDetailed",
    "ExampleMetric",
]

# Dictionary of available metrics for easy lookup
available_metrics = {
    "rmse": RMSE,
    "rmse_aggregate": RMSEAggregate,
    "mae": MAE,
    "mae_aggregate": MAEAggregate,
    "detailed_rmse": DetailedRMSE,
    "detailed_crps": DetailedCRPS,
    "crps_per_location": CRPSPerLocation,
    "crps": CRPS,
    "detailed_crps_norm": DetailedCRPSNorm,
    "crps_norm": CRPSNorm,
    #    "peak_value_diff": PeakValueDiffMetric,
    #    "peak_period_lag": PeakPeriodLagMetric,
    "samples_above_truth": SamplesAboveTruth,
    "is_within_10th_90th_detailed": IsWithin10th90thDetailed,
    "is_within_25th_75th_detailed": IsWithin25th75thDetailed,
    "ratio_within_10th_90th_per_location": RatioWithin10th90thPerLocation,
    "ratio_within_10th_90th": RatioWithin10th90th,
    "ratio_within_25th_75th_per_location": RatioWithin25th75thPerLocation,
    "ratio_within_25th_75th": RatioWithin25th75th,
    "test_sample_count_detailed": TestMetricDetailed,
    "test_sample_count": TestMetric,
    "example_metric": ExampleMetric,
}


def compute_all_aggregated_metrics_from_backtest(backtest: BackTest) -> dict[str, float]:
    relevant_metrics = {id: metric for id, metric in available_metrics.items() if metric().is_full_aggregate()}
    logger.info(f"Relevant metrics for aggregation: {relevant_metrics.keys()}")

    # Use Evaluation abstraction to get flat representation
    evaluation = Evaluation.from_backtest(backtest)
    flat_data = evaluation.to_flat()

    results = {}
    for id, metric_cls in relevant_metrics.items():
        metric = metric_cls()
        metric_df = metric.get_metric(flat_data.observations, flat_data.forecasts)
        if len(metric_df) != 1:
            raise ValueError(
                f"Metric {id} was expected to return a single aggregated value, but got {len(metric_df)} rows."
            )
        results[id] = float(metric_df["metric"].iloc[0])

    return results
