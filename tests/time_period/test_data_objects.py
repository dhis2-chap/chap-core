import pytest

from climate_health.time_period import TimePeriod, Month, Day, get_number_of_days


def test_time_period_month():
    month = TimePeriod.from_string('2013-01')
    assert isinstance(month, Month)
    assert str(month) == 'January 2013'


def test_time_period_day():
    day = TimePeriod.from_string('2013-01-02')
    assert isinstance(day, Day)
    assert day.year == 2013
    assert day.month == 1
    assert day.day == 2


@pytest.mark.skip
def test_number_of_days():
    assert get_number_of_days(2013, 1) == 31
    assert get_number_of_days(2013, 2) == 28
    assert get_number_of_days(2004, 2) == 29
    assert get_number_of_days(2000, 2) == 29
    assert get_number_of_days(1900, 2) == 28

    # assert str(day) == '1. January 2013'
