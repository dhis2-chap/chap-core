import numpy as np

from .dataclasses import Period, Year, Month, Day, Week
from . import TimePeriod, Month as sMonth


def month_range(start_period: Month, end_period: Month):
    n_months = (end_period.year - start_period.year) * 12 + (end_period.month+1 - start_period.month)
    global_month = np.arange(start_period.month, start_period.month + n_months)
    return Month(year=global_month//12+start_period.year,
                 month=global_month % 12)


def period_range(start_period, end_period, exclusive_end=False):
    if isinstance(start_period, sMonth):
        return month_range(start_period, end_period)
    raise Exception('Cannot genereate range for period type: ' + str(type(start_period)) + ' to ' + str(type(end_period)))
