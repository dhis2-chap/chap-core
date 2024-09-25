import json
import numpy as np
import pytest

from chap_core.climate_data.seasonal_forecasts import SeasonalForecast
from chap_core.time_period.date_util_wrapper import PeriodRange


@pytest.fixture
def seasonal_forecast_mock(data_path):
    precipitaion_data = open(data_path / "precipitation_seasonal_forecast.json").read()
    temperature_data = open(data_path / "temperature_seasonal_forecast.json").read()

    forecaster = SeasonalForecast()
    forecaster.add_json("rainfall", json.loads(precipitaion_data))
    forecaster.add_json("mean_temperature", json.loads(temperature_data))
    return forecaster


def test_seasonal_forecast(seasonal_forecast_mock):
    period_range = PeriodRange.from_strings(["2021-03", "2021-04"])
    temp = seasonal_forecast_mock.get_forecasts(
        "fdc6uOvgoji", period_range, "mean_temperature"
    )
    assert np.allclose(temp.value, [22.550374349, 22.676574707])
