from typing import Union

import pandas as pd
from pandas._libs.tslibs import IncompatibleFrequency

from climate_health.time_period.protocols import TimeStamp


class TimePeriod(pd.Period):
    def __le__(self, other: Union['TimePeriod', TimeStamp]) -> bool:
        ...
