import numpy as np
import pytest

from chap_core.climate_predictor import (
    MonthlyClimatePredictor,
    QuickForecastFetcher,
    WeeklyClimatePredictor,
)
from chap_core.datatypes import ClimateData, HealthData
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.time_period import PeriodRange, Month, Week


@pytest.fixture
def climate_data():
    time_period = PeriodRange.from_time_periods(Month.parse("2020-01"), Month.parse("2020-12"))
    values = np.arange(len(time_period))
    return DataSet(
        {
            "oslo": ClimateData(time_period, values, values * 2, values * 3),
            "stockholm": ClimateData(time_period, values, values * 2, values * 3),
        }
    )


@pytest.fixture
def weekly_climate_data():
    time_period = PeriodRange.from_time_periods(Week(2019, 1), Week(2020, 52))
    values = np.arange(len(time_period))
    return DataSet(
        {
            "oslo": ClimateData(time_period, values, values * 2, values * 3),
            "stockholm": ClimateData(time_period, values, values * 2, values * 3),
        }
    )


def test_climate_predictor(climate_data):
    predictor = MonthlyClimatePredictor()
    predictor.train(climate_data)
    time_period = PeriodRange.from_time_periods(Month.parse("2021-01"), Month.parse("2021-12"))
    prediction = predictor.predict(time_period)


def test_weekly_climate_predictor(weekly_climate_data):
    predictor = WeeklyClimatePredictor()
    predictor.train(weekly_climate_data)
    time_period = PeriodRange.from_time_periods(Week(2021, 1), Week(2021, 52))
    prediction = predictor.predict(time_period)


@pytest.mark.parametrize("fixture_name", ["climate_data", "weekly_climate_data"])
def test_target_only_future_preserves_locations_and_periods(request, fixture_name):
    climate = request.getfixturevalue(fixture_name)
    history = DataSet({location: HealthData(data.time_period, data.rainfall) for location, data in climate.items()})
    periods = climate.period_range[-3:]

    future = QuickForecastFetcher(history).get_future_weather(periods)

    assert set(future.keys()) == set(history.keys())
    assert future.field_names() == []
    for data in future.values():
        assert list(map(str, data.time_period)) == list(map(str, periods))
