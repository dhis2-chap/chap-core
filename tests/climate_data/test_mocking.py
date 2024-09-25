from climate_health.datatypes import Location
from climate_health.time_period import Month
from ..mocks import ClimateDataBaseMock


def test_mock():
    location = Location(100.0, 100.0)
    start_month = Month(2012, 1)
    mocked_data = ClimateDataBaseMock().get_data(location, start_month, Month(2012, 7))
    assert len(mocked_data) == 7
    mocked_data = ClimateDataBaseMock().get_data(location, start_month, Month(2013, 7))
    assert len(mocked_data) == 19
