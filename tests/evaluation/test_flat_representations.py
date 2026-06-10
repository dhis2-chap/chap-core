import pytest
from chap_core.assessment.flat_representations import (
    convert_backtest_observations_to_flat_observations,
    convert_backtest_to_flat_forecasts,
    max_horizon_distance,
)


def test_convert_backtest_to_flat_forecast(backtest_weeks_large):
    """
    Test converting BacktestForecast objects to ForecastFlatDataSchema format.

    This test uses the same backtest fixture as test_external_evaluation
    and prints the resulting flat DataFrame.
    """
    backtest = backtest_weeks_large
    flat_forecasts = convert_backtest_to_flat_forecasts(backtest.forecasts)
    flat_observations = convert_backtest_observations_to_flat_observations(backtest.dataset.observations)

    print(flat_forecasts)
    print(flat_observations)


def test_max_horizon_distance(forecasts):
    assert max_horizon_distance(forecasts) == 4


def test_max_horizon_distance_weekly(forecasts_weeks):
    assert max_horizon_distance(forecasts_weeks) == 4


def test_max_horizon_distance_empty():
    assert max_horizon_distance([]) is None
