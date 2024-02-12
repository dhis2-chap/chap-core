from typing import Protocol

from climate_health.datatypes import ClimateData
from climate_health.time_period import TimePeriod


class ClimateDataBase(Protocol):
    def get_data(self, region: str, start_period: TimePeriod, end_period: TimePeriod, exclusive_end=True) -> ClimateData:
        ...

