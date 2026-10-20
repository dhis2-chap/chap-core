from chap_core.assessment.flat_representations import (
    convert_backtest_observations_to_flat_observations,
    convert_backtest_to_flat_forecasts,
    max_horizon_distance,
)


def test_convert_backtest_to_flat_forecast(backtest_weeks_large):
    """
    Test converting BacktestForecast objects to ForecastFlatDataSchema format.

    """
    backtest = backtest_weeks_large
    flat_forecasts = convert_backtest_to_flat_forecasts(backtest.forecasts)
    flat_observations = convert_backtest_observations_to_flat_observations(backtest.dataset.observations)

    assert list(flat_forecasts.columns) == ["location", "time_period", "horizon_distance", "sample", "forecast"]
    assert list(flat_observations.columns) == ["location", "time_period", "disease_cases"]
    assert len(flat_observations) == len(backtest.dataset.observations)


def test_max_horizon_distance(forecasts):
    # Furthest forecast: period 2022-02 seen from 2021-11 -> 3 months -> horizon 4
    assert max_horizon_distance(forecasts) == 4


def test_max_horizon_distance_weekly(forecasts_weeks):
    # Furthest forecast: period 2022W02 seen from 2021W51 -> 3 weeks -> horizon 4
    assert max_horizon_distance(forecasts_weeks) == 4


def test_max_horizon_distance_empty():
    assert max_horizon_distance([]) is None
