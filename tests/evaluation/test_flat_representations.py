import pytest
from chap_core.assessment.flat_representations import (
    convert_backtest_observations_to_flat_observations,
    convert_backtest_to_flat_forecasts,
)


def test_convert_backtest_to_flat_forecast(backtest_weeks_large):
    """
    Test converting BackTestForecast objects to ForecastFlatDataSchema format.

    This test uses the same backtest fixture as test_external_evaluation
    and prints the resulting flat DataFrame.
    """
    backtest = backtest_weeks_large
    flat_forecasts = convert_backtest_to_flat_forecasts(backtest.forecasts)
    flat_observations = convert_backtest_observations_to_flat_observations(backtest.dataset.observations)

    print(flat_forecasts)
    print(flat_observations)
