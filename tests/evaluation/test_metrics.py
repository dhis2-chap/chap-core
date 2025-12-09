import numpy as np
from sqlalchemy import create_engine
from sqlmodel import SQLModel

from chap_core.assessment.metrics.rmse import RMSE, RMSEAggregate, DetailedRMSE
from chap_core.assessment.metrics.mae import MAE, MAEAggregate
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


@pytest.mark.skip(reason="Only for testing")
def test_read_example_weekly_predictions(data_path):
    data = pd.read_csv(data_path / "example_weekly_predictions.csv")
    data = SamplesWithTruth.from_pandas(data)
    print(type(data.time_period))
