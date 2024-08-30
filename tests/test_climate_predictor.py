import numpy as np
import pytest

from climate_health.climate_predictor import MonthlyClimatePredictor
from climate_health.datatypes import ClimateData
from climate_health.spatio_temporal_data.temporal_dataclass import DataSet
from climate_health.time_period import PeriodRange, Month


@pytest.fixture
def climate_data():
    time_period = PeriodRange.from_time_periods(Month.parse('2020-01'), Month.parse('2020-12'))
    values = np.arange(len(time_period))
    return DataSet(
        {'oslo': ClimateData(time_period, values, values*2, values*3),
        'stockholm': ClimateData(time_period, values, values*2, values*3)})



def test_climate_predictor(climate_data):
    predictor = MonthlyClimatePredictor()
    predictor.train(climate_data)
    time_period = PeriodRange.from_time_periods(Month.parse('2021-01'), Month.parse('2021-12'))
    prediction = predictor.predict(time_period)



