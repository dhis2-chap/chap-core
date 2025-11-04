"""
Metrics submodule for assessment.
All metrics are imported here for backwards compatibility.
"""

import logging
from chap_core.assessment.flat_representations import (
    FlatForecasts,
    FlatObserved,
    convert_backtest_observations_to_flat_observations,
    convert_backtest_to_flat_forecasts,
)
from chap_core.database.tables import BackTest
from chap_core.assessment.metrics.base import MetricBase, MetricSpec
from chap_core.assessment.metrics.rmse import RMSE, DetailedRMSE
from chap_core.assessment.metrics.mae import MAE
from chap_core.assessment.metrics.crps import CRPS, CRPSPerLocation, DetailedCRPS
from chap_core.assessment.metrics.crps_norm import CRPSNorm, DetailedCRPSNorm
from chap_core.assessment.metrics.peak_diff import PeakValueDiffMetric, PeakWeekLagMetric
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
    "DetailedRMSE",
    "MAE",
    "CRPS",
    "CRPSPerLocation",
    "DetailedCRPS",
    "CRPSNorm",
    "DetailedCRPSNorm",
    "PeakValueDiffMetric",
    "PeakWeekLagMetric",
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
    "mae": MAE,
    "detailed_rmse": DetailedRMSE,
    "detailed_crps": DetailedCRPS,
    "crps_per_location": CRPSPerLocation,
    "crps": CRPS,
    "detailed_crps_norm": DetailedCRPSNorm,
    "crps_norm": CRPSNorm,
    #    "peak_value_diff": PeakValueDiffMetric,
    #    "peak_week_lag": PeakWeekLagMetric,
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

    # Convert to flat representation
    flat_forecasts = FlatForecasts(convert_backtest_to_flat_forecasts(backtest.forecasts))
    flat_observations = FlatObserved(convert_backtest_observations_to_flat_observations(backtest.dataset.observations))

    results = {}
    for id, metric_cls in relevant_metrics.items():
        metric = metric_cls()
        metric_df = metric.get_metric(flat_observations, flat_forecasts)
        if len(metric_df) != 1:
            raise ValueError(
                f"Metric {id} was expected to return a single aggregated value, but got {len(metric_df)} rows."
            )
        results[id] = float(metric_df["metric"].iloc[0])

    return results
