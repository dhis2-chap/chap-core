import pytest

from climate_health.dataset import SpatioTemporalDict
from climate_health.datatypes import ClimateHealthData, ClimateData, HealthData
from climate_health.time_period import Month
from climate_health.time_period.period_range import period_range


@pytest.fixture()
def full_data() -> SpatioTemporalDict[ClimateHealthData]:
    time_period = period_range(Month(2012, 1), Month(2012, 12))
    T = len(time_period)
    d = {'oslo': ClimateHealthData(time_period, [1] * T, [1] * T, [20] * T),
         'bergen': ClimateHealthData(time_period, [100] * T, [1] * T, [1] * T)}
    return SpatioTemporalDict(d)


@pytest.fixture()
def train_data(full_data) -> SpatioTemporalDict[ClimateHealthData]:
    time_period = period_range(Month(2012, 1), Month(2012, 7))
    T = len(time_period)
    d = {'oslo': ClimateHealthData(time_period, [1] * T, [1] * T, [20] * T),
         'bergen': ClimateHealthData(time_period, [100] * T, [1] * T, [1] * T)}
    return SpatioTemporalDict(d)


@pytest.fixture()
def future_climate_data() -> SpatioTemporalDict[ClimateData]:
    time_period = period_range(Month(2012, 8), Month(2012, 12))
    T = len(time_period)
    d = {'oslo': ClimateData(time_period, [20] * T, [1] * T, [1] * T),
         'bergen': ClimateData(time_period, [1] * T, [100] * T, [1] * T)}
    return SpatioTemporalDict(d)


@pytest.fixture()
def bad_predictions():
    time_period = period_range(Month(2012, 8), Month(2012, 8))
    T = len(time_period)
    d = {'oslo': HealthData(time_period, [2] * T),
         'bergen': HealthData(time_period, [19] * T)}
    return SpatioTemporalDict(d)


@pytest.fixture()
def good_predictions():
    time_period = period_range(Month(2012, 8), Month(2012, 8))
    T = len(time_period)
    d = {'oslo': HealthData(time_period, [19] * T),
         'bergen': HealthData(time_period, [2] * T)}
    return SpatioTemporalDict(d)
