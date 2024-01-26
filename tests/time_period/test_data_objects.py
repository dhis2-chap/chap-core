import pytest

from climate_health.time_period import TimePeriod, Month


@pytest.mark.xfail
def test_time_period():
    month = TimePeriod.from_str('2013-01')
    assert str(month) == 'January 2013'