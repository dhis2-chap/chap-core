import numpy as np

from climate_health.datatypes import Shape, Location, ClimateData
from climate_health.time_period import TimePeriod, Month as sMonth
from climate_health.time_period.dataclasses import Month
from climate_health.time_period.period_range import period_range


class ClimateDataBaseMock:
    def get_data(self, region: Shape, start_period: TimePeriod, end_period, exclusive_end=False):
        assert isinstance(start_period, sMonth) and isinstance(end_period, sMonth)
        assert isinstance(region, Location)
        assert not exclusive_end
        period = period_range(start_period, end_period)
        # generate periodic monthly temperature
        temperature = 20 + 5 * np.sin(2 * np.pi * period.month / 12)
        rainfall = 100 + 50 * np.sin(2 * np.pi * period.month / 12)
        return ClimateData(period, rainfall, temperature, temperature)

