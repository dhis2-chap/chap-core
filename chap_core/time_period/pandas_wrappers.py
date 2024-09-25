from typing import Union

import pandas as pd

from chap_core.time_period.protocols import TimeStamp


class TimePeriod(pd.Period):
    def __le__(self, other: Union["TimePeriod", TimeStamp]) -> bool: ...
