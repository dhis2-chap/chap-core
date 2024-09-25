from typing import Protocol

from chap_core.datatypes import ClimateData, Shape
from chap_core.time_period import TimePeriod


class IsClimateDataBase(Protocol):
    def get_data(
        self,
        region: Shape,
        start_period: TimePeriod,
        end_period: TimePeriod,
        exclusive_end=True,
    ) -> ClimateData: ...
