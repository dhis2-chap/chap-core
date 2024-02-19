from climate_health.climate_data.gee import ERA5DataBase
from climate_health.datatypes import Location
from climate_health.time_period import Month
import pytest


@pytest.mark.skip
def test_era5():
    location = Location(17.9640988, 102.6133707)
    start_month = Month(2012, 1)
    mocked_data = ERA5DataBase().get_data(location, start_month, Month(2012, 7))
    assert len(mocked_data) == 7
    mocked_data = ERA5DataBase().get_data(location, start_month, Month(2013, 7))
    assert len(mocked_data) == 19
    full_data = ERA5DataBase().get_data(location, Month(2010, 1), Month(2024, 1))
    full_data.to_csv('climate_data.csv')