import numpy as np
from sqlalchemy import create_engine
from sqlmodel import SQLModel

from chap_core.assessment.metrics.rmse import RMSE, DetailedRMSE
from chap_core.assessment.metrics.peak_diff import PeakValueDiffMetric, PeakPeriodLagMetric
from chap_core.assessment.metrics import compute_all_aggregated_metrics_from_backtest
from chap_core.assessment.flat_representations import FlatForecasts, FlatObserved
import pytest
import pandas as pd

from chap_core.database.database import SessionWrapper
from chap_core.database.tables import BackTest
from chap_core.datatypes import SamplesWithTruth


@pytest.fixture
def test_flat_observed_with_nan():
    return FlatObserved(
        pd.DataFrame(
            {
                "location": ["loc1", "loc1", "loc2", "loc2"],
                "time_period": ["2023-W01", "2023-W02", "2023-W01", "2023-W02"],
                "disease_cases": [11.0, None, np.nan, 21.0],
            }
        )
    )


def test_rmse(flat_forecasts, flat_observations):
    rmse = RMSE()
    result = rmse.compute(flat_observations, flat_forecasts)

    correct = pd.DataFrame({"location": ["loc1", "loc2"], "metric": [1.0, 2.0]})

    pd.testing.assert_frame_equal(
        result.sort_values("location").reset_index(drop=True), correct.sort_values("location").reset_index(drop=True)
    )


def test_rmse_detailed(flat_forecasts, flat_observations):
    detailed_rmse = DetailedRMSE()
    result = detailed_rmse.compute(flat_observations, flat_forecasts)

    correct = pd.DataFrame(
        {
            "location": ["loc1", "loc1", "loc2", "loc2"],
            "time_period": ["2023-W01", "2023-W02", "2023-W01", "2023-W02"],
            "horizon_distance": [1, 2, 1, 2],
            "metric": [1.0, 1.0, 2.0, 2.0],
        }
    )

    pd.testing.assert_frame_equal(
        result.sort_values(["location", "horizon_distance"]).reset_index(drop=True),
        correct.sort_values(["location", "horizon_distance"]).reset_index(drop=True),
    )


@pytest.fixture
def flat_observations_week():
    """
    Observations with a clear peak per location.

    loc1: peak at 2023-W02, value = 20
    loc2: peak at 2023-W03, value = 9
    """
    obs_df = pd.DataFrame(
        {
            "location": ["loc1", "loc1", "loc1", "loc2", "loc2", "loc2"],
            "time_period": [
                "2023-W01",
                "2023-W02",
                "2023-W03",
                "2023-W01",
                "2023-W02",
                "2023-W03",
            ],
            "disease_cases": [10.0, 20.0, 15.0, 5.0, 7.0, 9.0],
        }
    )
    return FlatObserved(obs_df)


@pytest.fixture
def flat_forecasts_week():
    """
    Forecasts with one horizon where we control the forecast peak.

    horizon_distance = 1 for all.

    loc1 forecast_mean by week:
        W01: 8
        W02: 18 (peak)
        W03: 12

    loc2 forecast_mean by week:
        W01: 4
        W02: 10 (peak)
        W03: 6
    """
    fc_df = pd.DataFrame(
        {
            "location": ["loc1", "loc1", "loc1", "loc2", "loc2", "loc2"],
            "time_period": [
                "2023-W01",
                "2023-W02",
                "2023-W03",
                "2023-W01",
                "2023-W02",
                "2023-W03",
            ],
            "horizon_distance": [1, 1, 1, 1, 1, 1],
            "sample": [0, 0, 0, 0, 0, 0],
            "forecast": [8.0, 18.0, 12.0, 4.0, 10.0, 6.0],
        }
    )
    return FlatForecasts(fc_df)


def test_peak_value_diff_metric_weekly(flat_observations_week, flat_forecasts_week):
    """
    PeakValueDiffMetric should return:
        metric = truth_peak_value - pred_peak_value, per location & horizon.

    For the fixtures:

    loc1:
        truth peak: 20 (at 2023-W02)
        pred peak:  18 (at 2023-W02)
        => metric = 20 - 18 = 2

    loc2:
        truth peak:  9 (at 2023-W03)
        pred peak:  10 (at 2023-W02)
        => metric = 9 - 10 = -1
    """
    metric = PeakValueDiffMetric()
    result = metric.compute(flat_observations_week, flat_forecasts_week)

    # sort for stable comparison
    result_sorted = result.sort_values(["location", "horizon_distance"]).reset_index(drop=True)

    expected = pd.DataFrame(
        {
            "location": ["loc1", "loc2"],
            "time_period": ["2023-W02", "2023-W03"],  # truth peak time
            "horizon_distance": [1, 1],
            "metric": [2.0, -1.0],
        }
    )

    expected_sorted = expected.sort_values(["location", "horizon_distance"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(result_sorted, expected_sorted)


def test_peak_period_lag_metric_weekly(flat_observations_week, flat_forecasts_week):
    """
    PeakPeriodLagMetric should return:
        metric = time_index(pred_peak) - time_index(truth_peak)

    With weekly strings, that corresponds to:
    - same week -> 0
    - one week earlier than truth -> -1
    - one week later than truth -> +1

    For our fixtures:

    loc1:
        truth peak: 2023-W02
        pred peak:  2023-W02
        => lag = 0

    loc2:
        truth peak: 2023-W03
        pred peak:  2023-W02
        => lag = index(W02) - index(W03) = -1
    """
    metric = PeakPeriodLagMetric()
    result = metric.compute(flat_observations_week, flat_forecasts_week)

    result_sorted = result.sort_values(["location", "horizon_distance"]).reset_index(drop=True)

    expected = pd.DataFrame(
        {
            "location": ["loc1", "loc2"],
            "time_period": ["2023-W02", "2023-W03"],  # truth peak time
            "horizon_distance": [1, 1],
            "metric": [0.0, -1.0],
        }
    )

    expected_sorted = expected.sort_values(["location", "horizon_distance"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(result_sorted, expected_sorted)


def test_peak_metrics_get_metric_interface_weekly(flat_observations_week, flat_forecasts_week):
    """
    Check that MetricBase.get_metric works and enforces:
    - exactly one numeric *metric* column ('metric') beyond any numeric dimension columns
    - output matches compute(..)
    """

    for metric_cls in (PeakValueDiffMetric, PeakPeriodLagMetric):
        metric = metric_cls()

        # compute via compute()
        df_compute = metric.compute(flat_observations_week, flat_forecasts_week)

        # compute via get_metric() (which validates spec & numeric columns)
        df_get = metric.get_metric(flat_observations_week, flat_forecasts_week)

        # should be identical in content
        df_compute_sorted = df_compute.sort_values(df_compute.columns.tolist()).reset_index(drop=True)
        df_get_sorted = df_get.sort_values(df_get.columns.tolist()).reset_index(drop=True)

        pd.testing.assert_frame_equal(df_compute_sorted, df_get_sorted)

        # --- NEW: only enforce that there is exactly one *non-dimension* numeric column: 'metric'
        # Get the column names for the dimensions from the MetricSpec
        dim_cols = [dim.value for dim in metric.spec.output_dimensions]

        # All numeric columns in the result
        numeric_cols = df_get.select_dtypes(include=["number"]).columns.tolist()

        # Numeric columns that are NOT dimensions = metric columns
        metric_numeric_cols = [c for c in numeric_cols if c not in dim_cols]

        assert metric_numeric_cols == ["metric"], (
            f"{metric_cls.__name__} returned wrong numeric metric columns: {metric_numeric_cols} "
            f"(dimension columns are allowed to be numeric: {dim_cols})"
        )


@pytest.fixture
def flat_observations_monthly():
    """
    Observations with a clear monthly peak per location.

    loc1: peak at 2023-02, value = 20
    loc2: peak at 2023-03, value = 9
    """
    obs_df = pd.DataFrame(
        {
            "location": ["loc1", "loc1", "loc1", "loc2", "loc2", "loc2"],
            "time_period": [
                "2023-01",
                "2023-02",
                "2023-03",
                "2023-01",
                "2023-02",
                "2023-03",
            ],
            "disease_cases": [10.0, 20.0, 15.0, 5.0, 7.0, 9.0],
        }
    )
    return FlatObserved(obs_df)


@pytest.fixture
def flat_forecasts_monthly():
    """
    Forecasts with one horizon where we control the forecast peak.

    horizon_distance = 1 for all.

    loc1 forecast_mean by month:
        2023-01: 8
        2023-02: 18 (peak)
        2023-03: 12

    loc2 forecast_mean by month:
        2023-01: 4
        2023-02: 10 (peak)
        2023-03: 6
    """
    fc_df = pd.DataFrame(
        {
            "location": ["loc1", "loc1", "loc1", "loc2", "loc2", "loc2"],
            "time_period": [
                "2023-01",
                "2023-02",
                "2023-03",
                "2023-01",
                "2023-02",
                "2023-03",
            ],
            "horizon_distance": [1, 1, 1, 1, 1, 1],
            "sample": [0, 0, 0, 0, 0, 0],
            "forecast": [8.0, 18.0, 12.0, 4.0, 10.0, 6.0],
        }
    )
    return FlatForecasts(fc_df)


def test_peak_value_diff_metric_monthly(flat_observations_monthly, flat_forecasts_monthly):
    """
    PeakValueDiffMetric (monthly) should return:
        metric = truth_peak_value - pred_peak_value, per location & horizon.

    For the fixtures:

    loc1:
        truth peak: 20 at 2023-02
        pred peak:  18 at 2023-02
        => metric = 20 - 18 = 2

    loc2:
        truth peak:  9 at 2023-03
        pred peak:  10 at 2023-02
        => metric = 9 - 10 = -1
    """
    metric = PeakValueDiffMetric()
    result = metric.compute(flat_observations_monthly, flat_forecasts_monthly)

    result_sorted = result.sort_values(["location", "horizon_distance"]).reset_index(drop=True)

    expected = pd.DataFrame(
        {
            "location": ["loc1", "loc2"],
            "time_period": ["2023-02", "2023-03"],  # truth peak month
            "horizon_distance": [1, 1],
            "metric": [2.0, -1.0],
        }
    )
    expected_sorted = expected.sort_values(["location", "horizon_distance"]).reset_index(drop=True)

    pd.testing.assert_frame_equal(result_sorted, expected_sorted)


def test_peak_period_lag_metric_monthly(flat_observations_monthly, flat_forecasts_monthly):
    """
    PeakPeriodLagMetric (monthly) should return:
        metric = time_index(pred_peak) - time_index(truth_peak)

    With monthly strings and _time_index logic:
      index = year * 12 + (month - 1)

    So difference of one month earlier = -1, one month later = +1.

    For our fixtures:

    loc1:
        truth peak: 2023-02
        pred peak:  2023-02
        => lag = 0

    loc2:
        truth peak: 2023-03
        pred peak:  2023-02
        => lag = index(2023-02) - index(2023-03) = -1
    """
    metric = PeakPeriodLagMetric()
    result = metric.compute(flat_observations_monthly, flat_forecasts_monthly)

    result_sorted = result.sort_values(["location", "horizon_distance"]).reset_index(drop=True)

    expected = pd.DataFrame(
        {
            "location": ["loc1", "loc2"],
            "time_period": ["2023-02", "2023-03"],  # truth peak month
            "horizon_distance": [1, 1],
            "metric": [0.0, -1.0],
        }
    )
    expected_sorted = expected.sort_values(["location", "horizon_distance"]).reset_index(drop=True)

    pd.testing.assert_frame_equal(result_sorted, expected_sorted)


def test_peak_metrics_get_metric_interface_monthly(flat_observations_monthly, flat_forecasts_monthly):
    """
    Check that MetricBase.get_metric works on monthly data and enforces:
    - exactly one numeric *metric* column ('metric') beyond any numeric dimension columns
    - output matches compute(..)
    """
    for metric_cls in (PeakValueDiffMetric, PeakPeriodLagMetric):
        metric = metric_cls()

        df_compute = metric.compute(flat_observations_monthly, flat_forecasts_monthly)
        df_get = metric.get_metric(flat_observations_monthly, flat_forecasts_monthly)

        df_compute_sorted = df_compute.sort_values(df_compute.columns.tolist()).reset_index(drop=True)
        df_get_sorted = df_get.sort_values(df_get.columns.tolist()).reset_index(drop=True)

        pd.testing.assert_frame_equal(df_compute_sorted, df_get_sorted)

        # --- NEW: allow numeric dimension columns, only enforce metric column
        dim_cols = [dim.value for dim in metric.spec.output_dimensions]

        numeric_cols = df_get.select_dtypes(include=["number"]).columns.tolist()
        metric_numeric_cols = [c for c in numeric_cols if c not in dim_cols]

        assert metric_numeric_cols == ["metric"], (
            f"{metric_cls.__name__} returned wrong numeric metric columns: {metric_numeric_cols} "
            f"(dimension columns are allowed to be numeric: {dim_cols})"
        )



def test_get_all_aggregated_metrics_from_backtest(backtest_weeks):
    # write to and from csv first
    metrics = compute_all_aggregated_metrics_from_backtest(backtest_weeks)
    assert metrics["test_sample_count"] == 24.0


@pytest.fixture
def engine():
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    return engine


@pytest.mark.skip(reason="Only for testing")
def test_read_example_weekly_predictions(data_path):
    data = pd.read_csv(data_path / "example_weekly_predictions.csv")
    data = SamplesWithTruth.from_pandas(data)
    print(type(data.time_period))
