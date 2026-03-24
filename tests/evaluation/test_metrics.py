import numpy as np
from sqlalchemy import create_engine
from sqlmodel import SQLModel

from chap_core.assessment.metrics import (
    RMSEMetric,
    MAEMetric,
    CRPSMetric,
    CRPSLog1pMetric,
    CRPSNormMetric,
    RatioAboveTruthMetric,
    Coverage10_90Metric,
    Coverage25_75Metric,
    SampleCountMetric,
    ExampleMetric,
    WinklerScore10_90Log1pMetric,
    WinklerScore10_90Metric,
    WinklerScore25_75Log1pMetric,
    WinklerScore25_75Metric,
    DataDimension,
    compute_all_aggregated_metrics_from_backtest,
)
from chap_core.assessment.metrics.peak_diff import PeakValueDiffMetric, PeakPeriodLagMetric
from chap_core.assessment.flat_representations import FlatForecasts, FlatObserved
import pytest
import pandas as pd

from chap_core.database.database import SessionWrapper
from chap_core.database.tables import BackTest
from chap_core.datatypes import SamplesWithTruth


# --- Parametrized tests for all metrics and all 3 methods ---

ALL_METRIC_FACTORIES = [
    RMSEMetric,
    MAEMetric,
    CRPSMetric,
    CRPSLog1pMetric,
    CRPSNormMetric,
    RatioAboveTruthMetric,
    Coverage10_90Metric,
    Coverage25_75Metric,
    SampleCountMetric,
    ExampleMetric,
    WinklerScore10_90Metric,
    WinklerScore10_90Log1pMetric,
    WinklerScore25_75Metric,
    WinklerScore25_75Log1pMetric,
]


@pytest.mark.parametrize("metric_factory", ALL_METRIC_FACTORIES)
def test_get_global_metric_returns_single_row(metric_factory, flat_forecasts_multiple_samples, flat_observations):
    """Test that get_global_metric() returns a single-row DataFrame for all metrics."""
    metric = metric_factory()
    result = metric.get_global_metric(flat_observations, flat_forecasts_multiple_samples)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    assert "metric" in result.columns
    assert result.columns.tolist() == ["metric"]


@pytest.mark.parametrize("metric_factory", ALL_METRIC_FACTORIES)
def test_get_detailed_metric_returns_all_dimensions(metric_factory, flat_forecasts_multiple_samples, flat_observations):
    """Test that get_detailed_metric() returns DataFrame with all output dimensions."""
    metric = metric_factory()
    result = metric.get_detailed_metric(flat_observations, flat_forecasts_multiple_samples)

    assert isinstance(result, pd.DataFrame)
    assert "metric" in result.columns
    # Check that all output dimensions are present
    for dim in metric.spec.output_dimensions:
        assert dim.value in result.columns


@pytest.mark.parametrize("metric_factory", ALL_METRIC_FACTORIES)
def test_get_metric_with_location_dimension(metric_factory, flat_forecasts_multiple_samples, flat_observations):
    """Test that get_metric() with location dimension returns per-location results."""
    metric = metric_factory()
    result = metric.get_metric(flat_observations, flat_forecasts_multiple_samples, dimensions=(DataDimension.location,))

    assert isinstance(result, pd.DataFrame)
    assert "metric" in result.columns
    assert "location" in result.columns
    assert result.columns.tolist() == ["location", "metric"]
    # Should have one row per location
    assert len(result) == flat_observations["location"].nunique()


@pytest.mark.parametrize("metric_factory", ALL_METRIC_FACTORIES)
def test_get_metric_with_horizon_dimension(metric_factory, flat_forecasts_multiple_samples, flat_observations):
    """Test that get_metric() with horizon dimension returns per-horizon results."""
    metric = metric_factory()
    result = metric.get_metric(
        flat_observations, flat_forecasts_multiple_samples, dimensions=(DataDimension.horizon_distance,)
    )

    assert isinstance(result, pd.DataFrame)
    assert "metric" in result.columns
    assert "horizon_distance" in result.columns
    assert result.columns.tolist() == ["horizon_distance", "metric"]


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


def test_rmse_per_location(flat_forecasts, flat_observations):
    """Test RMSEMetric with per-location dimension."""
    rmse = RMSEMetric()
    result = rmse.get_metric(flat_observations, flat_forecasts, dimensions=(DataDimension.location,))

    correct = pd.DataFrame({"location": ["loc1", "loc2"], "metric": [1.0, 2.0]})

    pd.testing.assert_frame_equal(
        result.sort_values("location").reset_index(drop=True), correct.sort_values("location").reset_index(drop=True)
    )


def test_rmse_detailed(flat_forecasts, flat_observations):
    """Test RMSEMetric at detailed level using get_detailed_metric()."""
    rmse = RMSEMetric()
    result = rmse.get_detailed_metric(flat_observations, flat_forecasts)

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
    """Test RMSEMetric at global level computes a single value across all data."""
    rmse = RMSEMetric()
    result = rmse.get_global_metric(flat_observations, flat_forecasts)

    assert len(result) == 1
    assert "metric" in result.columns

    # Manually compute expected: errors are [1, 1, 2, 2], squared = [1, 1, 4, 4]
    # MSE = (1+1+4+4)/4 = 2.5, RMSE = sqrt(2.5) ~ 1.58
    expected_rmse = np.sqrt(2.5)
    np.testing.assert_almost_equal(result["metric"].iloc[0], expected_rmse, decimal=5)


def test_mae_aggregate(flat_forecasts, flat_observations):
    """Test MAEMetric at global level computes a single value across all data."""
    mae = MAEMetric()
    result = mae.get_global_metric(flat_observations, flat_forecasts)

    assert len(result) == 1
    assert "metric" in result.columns

    # Manually compute expected: errors are [1, 1, 2, 2]
    # MAE = (1+1+2+2)/4 = 1.5
    expected_mae = 1.5
    np.testing.assert_almost_equal(result["metric"].iloc[0], expected_mae, decimal=5)


def test_get_all_aggregated_metrics_from_backtest(backtest_weeks):
    """Test compute_all_aggregated_metrics_from_backtest returns expected metrics."""
    metrics = compute_all_aggregated_metrics_from_backtest(backtest_weeks)
    assert "sample_count" in metrics
    assert metrics["sample_count"] == 24.0


@pytest.fixture
def engine():
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    return engine


def test_rmse_uses_median_of_samples(flat_forecasts_multiple_samples, flat_observations):
    """Test that RMSEMetric computes errors from median of samples, not per-sample."""
    rmse = RMSEMetric()
    result = rmse.get_metric(flat_observations, flat_forecasts_multiple_samples, dimensions=(DataDimension.location,))

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
    """Test that RMSEMetric at global level computes errors from median of samples."""
    rmse = RMSEMetric()
    result = rmse.get_global_metric(flat_observations, flat_forecasts_multiple_samples)

    # Median forecasts: [10, 12, 21, 23]
    # Observations: [11, 13, 19, 21]
    # Errors: [1, 1, 2, 2]
    # Squared errors: [1, 1, 4, 4]
    # MSE = (1+1+4+4)/4 = 2.5
    # RMSE = sqrt(2.5)
    expected_rmse = np.sqrt(2.5)
    np.testing.assert_almost_equal(result["metric"].iloc[0], expected_rmse, decimal=5)


def test_detailed_rmse_uses_median_of_samples(flat_forecasts_multiple_samples, flat_observations):
    """Test that RMSEMetric at detailed level computes errors from median of samples."""
    rmse = RMSEMetric()
    result = rmse.get_detailed_metric(flat_observations, flat_forecasts_multiple_samples)

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
    """Test that MAEMetric computes errors from median of samples."""
    mae = MAEMetric()
    result = mae.get_detailed_metric(flat_observations, flat_forecasts_multiple_samples)

    # Median forecasts: loc1=[10, 12] (horizons 1, 2), loc2=[21, 23] (horizons 1, 2)
    # Observations: loc1=[11, 13], loc2=[19, 21]
    # Absolute errors: loc1=[1, 1], loc2=[2, 2]
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


def test_mae_aggregate_uses_median_of_samples(flat_forecasts_multiple_samples, flat_observations):
    """Test that MAEMetric at global level computes errors from median of samples."""
    mae = MAEMetric()
    result = mae.get_global_metric(flat_observations, flat_forecasts_multiple_samples)

    # Median forecasts: [10, 12, 21, 23]
    # Observations: [11, 13, 19, 21]
    # Absolute errors: [1, 1, 2, 2]
    # MAE = (1+1+2+2)/4 = 1.5
    expected_mae = 1.5
    np.testing.assert_almost_equal(result["metric"].iloc[0], expected_mae, decimal=5)


@pytest.mark.skip(reason="Only for testing")
def test_read_example_weekly_predictions(data_path):
    data = pd.read_csv(data_path / "example_weekly_predictions.csv")
    samples = SamplesWithTruth.from_pandas(data)
    print(type(samples.time_period))


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
    result = metric.get_detailed_metric(flat_observations_week, flat_forecasts_week)

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
    result = metric.get_detailed_metric(flat_observations_week, flat_forecasts_week)

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
    result = metric.get_detailed_metric(flat_observations_monthly, flat_forecasts_monthly)

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


def test_winkler_score_observation_inside_interval():
    """Test Winkler score when observation is inside the prediction interval."""
    samples = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0])
    observed = 50.0

    metric = WinklerScore25_75Metric()
    score = metric.compute_sample_metric(samples, observed)

    # 25th percentile = 25, 75th percentile = 75, interval width = 50
    # observation inside interval -> score = width = 50
    assert score == 50.0


def test_winkler_score_observation_below_interval():
    """Test Winkler score when observation falls below the prediction interval."""
    samples = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0])
    observed = 10.0

    metric = WinklerScore25_75Metric()
    score = metric.compute_sample_metric(samples, observed)

    # 25th percentile = 25, 75th percentile = 75, alpha = 0.5
    # score = 50 + (2/0.5) * (25 - 10) = 50 + 4 * 15 = 110
    assert score == 110.0


def test_winkler_score_observation_above_interval():
    """Test Winkler score when observation falls above the prediction interval."""
    samples = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0])
    observed = 90.0

    metric = WinklerScore25_75Metric()
    score = metric.compute_sample_metric(samples, observed)

    # 25th percentile = 25, 75th percentile = 75, alpha = 0.5
    # score = 50 + (2/0.5) * (90 - 75) = 50 + 4 * 15 = 110
    assert score == 110.0


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
    result = metric.get_detailed_metric(flat_observations_monthly, flat_forecasts_monthly)

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
