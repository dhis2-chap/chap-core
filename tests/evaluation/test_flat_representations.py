import pytest
from chap_core.assessment.flat_representations import convert_to_forecast_flat_data


def test_convert_backtest_to_flat_forecast(backtest):
    """
    Test converting BackTestForecast objects to ForecastFlatDataSchema format.
    
    This test uses the same backtest fixture as test_external_evaluation
    and prints the resulting flat DataFrame.
    """
    flat_df = convert_to_forecast_flat_data(backtest.forecasts)
    
    print(flat_df)