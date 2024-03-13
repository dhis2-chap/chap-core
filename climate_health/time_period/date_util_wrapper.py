import functools
from dataclasses import dataclass
from datetime import datetime
from typing import Union

from dateutil.parser import parse
from dateutil.relativedelta import relativedelta


class DateUtilWrapper:
    _used_attributes = []

    def __init__(self, date: datetime):
        self._date = date

    def __getattr__(self, item: str):
        if item in self._used_attributes:
            return getattr(self._date, item)
        return super().__getattribute__(item)


class TimeStamp(DateUtilWrapper):
    _used_attributes = ['year', 'month', 'day', '__str__', '__repr__']

    def __init__(self, date: datetime):
        self._date = date

    @classmethod
    def parse(cls, text_repr: str):
        return cls(parse(text_repr))

    def __le__(self, other: 'TimeStamp'):
        return self._comparison(other, '__le__')

    def __ge__(self, other: 'TimeStamp'):
        return self._comparison(other, '__ge__')

    def __gt__(self, other: 'TimeStamp'):
        return self._comparison(other, '__gt__')

    def __lt__(self, other: 'TimeStamp'):
        return self._comparison(other, '__lt__')

    def _comparison(self, other: 'TimeStamp', func_name: str):
        return getattr(self._date, func_name)(other._date)


class TimePeriod:
    _used_attributes = []

    def __init__(self):
        self._date = None
        self._extension = None

    def __eq__(self, other):
        return (self._date == other._date) and (self._extension == other._extension)

    def __le__(self, other: 'TimePeriod'):
        if isinstance(other, TimeStamp):
            return TimeStamp(self._date) <= other

        return self._date < other._exclusive_end()

    def __ge__(self, other: 'TimePeriod'):
        if isinstance(other, TimeStamp):
            return TimeStamp(self._exclusive_end()) > other
        return self._exclusive_end() > other._date

    def __gt__(self, other: 'TimePeriod'):
        if isinstance(other, TimeStamp):
            return TimeStamp(self._date) > other
        return self._date >= other._exclusive_end()

    def __lt__(self, other: 'TimePeriod'):
        if isinstance(other, TimeStamp):
            return TimeStamp(self._exclusive_end()) <= other
        return self._exclusive_end() <= other._date

    def __repr__(self):
        return f'Month({self.year}-{self.month})'

    def _exclusive_end(self):
        return self._date + self._extension

    def __getattr__(self, item):
        if item in self._used_attributes:
            return getattr(self._date, item)
        return super().__getattribute__(item)

    @property
    def time_delta(self):
        return TimeDelta(self._extension)

    @classmethod
    def parse(cls, text_repr: str):
        default_dates = [datetime(2010, 1, 1), datetime(2009, 11, 10)]
        dates = [parse(text_repr, default=default_date) for default_date in default_dates]
        date = dates[0]
        if dates[0].day == dates[1].day:
            return Day(date)
        elif dates[0].month == dates[1].month:
            return Month(date)


class Day(TimePeriod):
    _used_attributes = ['year', 'month', 'day']

    def __init__(self, date):
        self._date = date
        self.year = date.year
        self.month = date.month
        self._extension = relativedelta(days=1)


class Month(TimePeriod):
    _used_attributes = ['year', 'month']

    def __init__(self, date: datetime):
        self._date = date
        self._extension = relativedelta(months=1)


class Year(TimePeriod):
    _used_attributes = ['year']

    def __init__(self, date: datetime):
        self._date = date
        self._extension = relativedelta(years=1)


class TimeDelta(DateUtilWrapper):
    def __init__(self, relative_delta: relativedelta):
        self._relative_delta = relative_delta
        self._date = None

    def __eq__(self, other):
        return self._relative_delta == other._relative_delta

    def __add__(self, other: Union[TimeStamp, TimePeriod]):
        if not isinstance(other, (TimeStamp, TimePeriod)):
            return NotImplemented
        return other.__class__(other._date + self._relative_delta)

    def __radd__(self, other: Union[TimeStamp, TimePeriod]):
        return self.__add__(other)

    def _n_months(self):
        return self._relative_delta.months + 12 * self._relative_delta.years

    def __floordiv__(self, divident: 'TimeDelta'):
        assert divident._relative_delta.days == 0
        return self._n_months() // divident._n_months()

    def __mod__(self, other: 'TimeDelta'):
        assert other._relative_delta.days == 0
        return self.__class__(relativedelta(months=self._n_months() % other._n_months()))


class PeriodRange:
    def __init__(self, start_period: TimePeriod, end_period: TimePeriod, inclusive=True):
        assert inclusive
        assert start_period.time_delta == end_period.time_delta
        self._start_period = start_period
        self._end_period = end_period
        self._time_delta = start_period.time_delta
        self._inclusive = inclusive

    def __len__(self):
        delta = relativedelta(
            self._end_period._exclusive_end(), # We consider the module the encapsulation here, could maybe be done more elegant
            self._start_period._date)
        return TimeDelta(delta) // self._time_delta



delta_month = TimeDelta(relativedelta(months=1))
delta_year = TimeDelta(relativedelta(years=1))
delta_day = TimeDelta(relativedelta(days=1))
