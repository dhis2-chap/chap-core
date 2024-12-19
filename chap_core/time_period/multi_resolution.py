import numpy as np

from chap_core.time_period import Month
from npstructures import RaggedArray


def pack_to_period(time_period, data, goal_period):
    time_period = time_period
    if goal_period is Month:
        changes = np.flatnonzero(np.diff(time_period.month)) + 1
        period_starts = np.insert(changes, 0, 0)
        new_index = time_period[period_starts]
        new_index = Month(month=new_index.month, year=new_index.year)
        period_lengths = np.diff(np.append(period_starts, len(time_period)))
        return new_index, RaggedArray(data, period_lengths)
