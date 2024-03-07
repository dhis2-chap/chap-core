import numpy as np

from .dataclasses import Period, Year, Month, Day, Week
from . import TimePeriod, Month as sMonth


def month_range(start_period: Month, end_period: Month, exclusive_end=False):
    n_months = (end_period.year - start_period.year) * 12 + (end_period.month+1 - start_period.month)
    if exclusive_end:
        n_months -= 1
    global_month = np.arange(start_period.month, start_period.month + n_months)
    return Month(year=global_month//12+start_period.year,
                 month=global_month % 12)


def period_range(start_period, end_period, exclusive_end=False):
    if hasattr(start_period, 'day') or hasattr(end_period, 'day'):
        raise NotImplementedError(f'Only monthly data is available, {start_period}, {end_period}')
    if not hasattr(start_period, 'month') or not hasattr(end_period, 'month'):
        raise NotImplementedError(f'Only monthly data is available, {start_period}, {end_period}')

    return month_range(start_period, end_period)

