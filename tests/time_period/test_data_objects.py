import pandas as pd
import pytest

from chap_core.time_period import TimePeriod, Month, Day
from chap_core.time_period import Year


def test_time_period_month():
    month = TimePeriod.parse("2013-01")
    assert isinstance(month, Month)
    assert str(month) == "Month(2013-1)"


def test_time_period_day():
    day = TimePeriod.parse("2013-01-02")
    assert isinstance(day, Day)
    assert day.year == 2013
    assert day.month == 1
    assert day.day == 2


@pytest.mark.skip("unused")
def test_number_of_days():
    assert get_number_of_days(Month(2013, 1)) == 31
    assert get_number_of_days(Month(2013, 2)) == 28
    assert get_number_of_days(Month(2004, 2)) == 29
    assert get_number_of_days(Month(2000, 2)) == 29
    assert get_number_of_days(Month(1900, 2)) == 28


@pytest.mark.skip("unused")
def test_topandas():
    years = Year([2013, 2014, 2015])
    series = years.topandas()
    true_series = pd.Series([pd.Period(year=y, freq="Y") for y in [2013, 2014, 2015]])
    assert series.equals(true_series)

    # assert str(day) == '1. January 2013'
