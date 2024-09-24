import numpy as np
import pandas as pd

from .dataclasses import Month, Day
from . import PeriodRange


def month_range(start_period: Month, end_period: Month, exclusive_end=False):
    n_months = (end_period.year - start_period.year) * 12 + (
        end_period.month + 1 - start_period.month
    )
    if exclusive_end:
        n_months -= 1
    global_month = np.arange(start_period.month, start_period.month + n_months)
    return Month(year=global_month // 12 + start_period.year, month=global_month % 12)


def get_n_days_in_year(year: int):
    if year % 4 != 0:
        return 365
    if year % 100 != 0:
        return 366
    if year % 400 != 0:
        return 365
    return 366


def get_n_days_in_month(month: int, year: int):
    month = month + 1
    if month in [1, 3, 5, 7, 8, 10, 12]:
        return 31
    if month in [4, 6, 9, 11]:
        return 30
    if month == 2:
        return 28 + get_n_days_in_year(year) - 365
    raise ValueError(f"Invalid month {month}")


def day_range(start_period, end_period, exclusive_end):
    pd_start = pd.Period(
        year=start_period.year,
        month=start_period.month + 1,
        day=start_period.day + 1,
        freq="D",
    )
    pd_end = pd.Period(
        year=end_period.year,
        month=end_period.month + 1,
        day=end_period.day + 1,
        freq="D",
    )
    period_range = pd.period_range(start=pd_start, end=pd_end, freq="D")
    return Day(
        year=period_range.year, month=period_range.month - 1, day=period_range.day - 1
    )


def period_range(start_period, end_period, exclusive_end=False):
    return PeriodRange.from_time_periods(start_period, end_period)
    # if hasattr(start_period, 'day') or hasattr(end_period, 'day'):
    #     return day_range(start_period, end_period, exclusive_end)
    # if not hasattr(start_period, 'month') or not hasattr(end_period, 'month'):
    #     raise NotImplementedError(f'Only monthly data is available, {start_period}, {end_period}')
    #
    # return month_range(start_period, end_period, exclusive_end)
