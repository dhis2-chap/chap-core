import pytest

from climate_health.time_period import TimePeriod, Month, Day


def test_time_period_month():
    month = TimePeriod.from_string('2013-01')
    assert isinstance(month, Month)
    assert str(month) == 'January 2013'


@pytest.mark.skip(reason='Not implemented yet')
def test_time_period_day():
    day = TimePeriod.from_string('2013-01-02')
    assert isinstance(day, Day)
    assert day.year == 2013
    assert day.month == 1
    assert day.day == 2
    # assert str(day) == '1. January 2013'
