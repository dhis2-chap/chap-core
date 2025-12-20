import numpy as np
from sqlalchemy import create_engine
from sqlmodel import SQLModel

from chap_core.assessment.metrics.rmse import RMSE, RMSEAggregate, DetailedRMSE
from chap_core.assessment.metrics.mae import MAE, MAEAggregate
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


def test_rmse_aggregate(flat_forecasts, flat_observations):
    """Test RMSEAggregate computes a single value across all data."""
    rmse_agg = RMSEAggregate()
    result = rmse_agg.compute(flat_observations, flat_forecasts)

    assert len(result) == 1
    assert "metric" in result.columns

    # Manually compute expected: errors are [1, 1, 2, 2], squared = [1, 1, 4, 4]
    # MSE = (1+1+4+4)/4 = 2.5, RMSE = sqrt(2.5) ~ 1.58
    expected_rmse = np.sqrt(2.5)
    np.testing.assert_almost_equal(result["metric"].iloc[0], expected_rmse, decimal=5)


def test_mae_aggregate(flat_forecasts, flat_observations):
    """Test MAEAggregate computes a single value across all data."""
    mae_agg = MAEAggregate()
    result = mae_agg.compute(flat_observations, flat_forecasts)

    assert len(result) == 1
    assert "metric" in result.columns

    # Manually compute expected: errors are [1, 1, 2, 2]
    # MAE = (1+1+2+2)/4 = 1.5
    expected_mae = 1.5
    np.testing.assert_almost_equal(result["metric"].iloc[0], expected_mae, decimal=5)


def test_rmse_aggregate_is_full_aggregate():
    """Test that RMSEAggregate.is_full_aggregate() returns True."""
    rmse_agg = RMSEAggregate()
    assert rmse_agg.is_full_aggregate() is True


def test_mae_aggregate_is_full_aggregate():
    """Test that MAEAggregate.is_full_aggregate() returns True."""
    mae_agg = MAEAggregate()
    assert mae_agg.is_full_aggregate() is True


def test_get_all_aggregated_metrics_from_backtest(backtest_weeks):
    # write to and from csv first
    metrics = compute_all_aggregated_metrics_from_backtest(backtest_weeks)
    assert metrics["test_sample_count"] == 24.0


@pytest.fixture
def engine():
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    return engine


def test_rmse_uses_median_of_samples(flat_forecasts_multiple_samples, flat_observations):
    """Test that RMSE computes errors from median of samples, not per-sample."""
    rmse = RMSE()
    result = rmse.compute(flat_observations, flat_forecasts_multiple_samples)

    # Median forecasts: loc1=[10, 12], loc2=[21, 23]
    # Observations: loc1=[11, 13], loc2=[19, 21]
    # Errors: loc1=[1, 1], loc2=[2, 2]
    # Squared errors: loc1=[1, 1], loc2=[4, 4]
    # MSE per location: loc1=1, loc2=4
    # RMSE per location: loc1=1.0, loc2=2.0
    correct = pd.DataFrame({"location": ["loc1", "loc2"], "metric": [1.0, 2.0]})

    pd.testing.assert_frame_equal(
        result.sort_values("location").reset_index(drop=True), correct.sort_values("location").reset_index(drop=True)
    )


def test_rmse_aggregate_uses_median_of_samples(flat_forecasts_multiple_samples, flat_observations):
    """Test that RMSEAggregate computes errors from median of samples."""
    rmse_agg = RMSEAggregate()
    result = rmse_agg.compute(flat_observations, flat_forecasts_multiple_samples)

    # Median forecasts: [10, 12, 21, 23]
    # Observations: [11, 13, 19, 21]
    # Errors: [1, 1, 2, 2]
    # Squared errors: [1, 1, 4, 4]
    # MSE = (1+1+4+4)/4 = 2.5
    # RMSE = sqrt(2.5)
    expected_rmse = np.sqrt(2.5)
    np.testing.assert_almost_equal(result["metric"].iloc[0], expected_rmse, decimal=5)


def test_detailed_rmse_uses_median_of_samples(flat_forecasts_multiple_samples, flat_observations):
    """Test that DetailedRMSE computes errors from median of samples."""
    detailed_rmse = DetailedRMSE()
    result = detailed_rmse.compute(flat_observations, flat_forecasts_multiple_samples)

    # Median forecasts per loc/time/horizon: [10, 12, 21, 23]
    # Observations: [11, 13, 19, 21]
    # Errors: [1, 1, 2, 2]
    # RMSE (absolute error for single point): [1, 1, 2, 2]
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


def test_mae_uses_median_of_samples(flat_forecasts_multiple_samples, flat_observations):
    """Test that MAE computes errors from median of samples."""
    mae = MAE()
    result = mae.compute(flat_observations, flat_forecasts_multiple_samples)

    # Median forecasts: loc1=[10, 12] (horizons 1, 2), loc2=[21, 23] (horizons 1, 2)
    # Observations: loc1=[11, 13], loc2=[19, 21]
    # Absolute errors: loc1=[1, 1], loc2=[2, 2]
    # MAE per location/horizon: loc1/h1=1, loc1/h2=1, loc2/h1=2, loc2/h2=2
    correct = pd.DataFrame(
        {
            "location": ["loc1", "loc1", "loc2", "loc2"],
            "horizon_distance": [1, 2, 1, 2],
            "metric": [1.0, 1.0, 2.0, 2.0],
        }
    )

    pd.testing.assert_frame_equal(
        result.sort_values(["location", "horizon_distance"]).reset_index(drop=True),
        correct.sort_values(["location", "horizon_distance"]).reset_index(drop=True),
    )


def test_mae_aggregate_uses_median_of_samples(flat_forecasts_multiple_samples, flat_observations):
    """Test that MAEAggregate computes errors from median of samples."""
    mae_agg = MAEAggregate()
    result = mae_agg.compute(flat_observations, flat_forecasts_multiple_samples)

    # Median forecasts: [10, 12, 21, 23]
    # Observations: [11, 13, 19, 21]
    # Absolute errors: [1, 1, 2, 2]
    # MAE = (1+1+2+2)/4 = 1.5
    expected_mae = 1.5
    np.testing.assert_almost_equal(result["metric"].iloc[0], expected_mae, decimal=5)


@pytest.mark.skip(reason="Only for testing")
def test_read_example_weekly_predictions(data_path):
    data = pd.read_csv(data_path / "example_weekly_predictions.csv")
    data = SamplesWithTruth.from_pandas(data)
    print(type(data.time_period))


# --- Peak diff metric tests ---


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

    result_sorted = result.sort_values(["location", "horizon_distance"]).reset_index(drop=True)

    expected = pd.DataFrame(
        {
            "location": ["loc1", "loc2"],
            "time_period": ["2023-W02", "2023-W03"],
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
            "time_period": ["2023-W02", "2023-W03"],
            "horizon_distance": [1, 1],
            "metric": [0.0, -1.0],
        }
    )

    expected_sorted = expected.sort_values(["location", "horizon_distance"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(result_sorted, expected_sorted)


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
            "time_period": ["2023-02", "2023-03"],
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
            "time_period": ["2023-02", "2023-03"],
            "horizon_distance": [1, 1],
            "metric": [0.0, -1.0],
        }
    )
    expected_sorted = expected.sort_values(["location", "horizon_distance"]).reset_index(drop=True)

    pd.testing.assert_frame_equal(result_sorted, expected_sorted)
