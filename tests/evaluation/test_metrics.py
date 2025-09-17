from chap_core.assessment.metrics import RMSE, DetailedRMSE, compute_all_aggregated_metrics_from_backtest
from chap_core.assessment.flat_representations import FlatForecasts, FlatObserved
import pytest
import pandas as pd


@pytest.fixture
def flat_forecasts():
    return FlatForecasts(pd.DataFrame({
        "location": ["loc1", "loc1", "loc2", "loc2"],
        "time_period": ["2023-W01", "2023-W02", "2023-W01", "2023-W02"],
        "horizon_distance": [1, 2, 1, 2],
        "sample": [1, 1, 1, 1],
        "forecast": [10, 12, 21, 23],
    }))

@pytest.fixture
def flat_observations():
    return FlatObserved(pd.DataFrame({
        "location": ["loc1", "loc1", "loc2", "loc2"],
        "time_period": ["2023-W01", "2023-W02", "2023-W01", "2023-W02"],
        "disease_cases": [11, 13, 19, 21],
    }))



def test_rmse(flat_forecasts, flat_observations):
    rmse = RMSE()
    result = rmse.compute(flat_observations, flat_forecasts)
    
    correct = pd.DataFrame({
        "location": ["loc1", "loc2"],
        "metric": [1.0, 2.0]
    })

    pd.testing.assert_frame_equal(result.sort_values("location").reset_index(drop=True),
                                    correct.sort_values("location").reset_index(drop=True))


def test_rmse_detailed(flat_forecasts, flat_observations):
    detailed_rmse = DetailedRMSE()
    result = detailed_rmse.compute(flat_observations, flat_forecasts)
    
    correct = pd.DataFrame({
        "location": ["loc1", "loc1", "loc2", "loc2"],
        "time_period": ["2023-W01", "2023-W02", "2023-W01", "2023-W02"],
        "horizon_distance": [1, 2, 1, 2],
        "metric": [1.0, 1.0, 2.0, 2.0]
    })

    pd.testing.assert_frame_equal(result.sort_values(["location", "horizon_distance"]).reset_index(drop=True),
                                    correct.sort_values(["location", "horizon_distance"]).reset_index(drop=True))


def test_get_all_aggregated_metrics_from_backtest(backtest):
    metrics = compute_all_aggregated_metrics_from_backtest(backtest)
    assert metrics["test_sample_count"] == 24.0


