import pandas as pd

from climate_health.climate_data.gee import ERA5DataBase, parse_gee_properties
from climate_health.datatypes import Location, ClimateData, SimpleClimateData
from climate_health.time_period import Month, Day, TimePeriod, PeriodRange
import pytest


@pytest.mark.skip('ee not supported')
def test_era5():
    location = Location(17.9640988, 102.6133707)
    start_month = Month(2012, 1)
    mocked_data = ERA5DataBase().get_data(location, start_month, Month(2012, 7))
    assert len(mocked_data) == 6
    mocked_data = ERA5DataBase().get_data(location, start_month, Month(2013, 7))
    assert len(mocked_data) == 18
    full_data = ERA5DataBase().get_data(location, Month(2010, 1), Month(2024, 1))
    full_data.to_csv('climate_data.csv')


@pytest.mark.skip('ee not supported')
def test_era5_daily():
    location = Location(17.9640988, 102.6133707)
    start_day = Day(2012, 1, 1)
    mocked_data = ERA5DataBase().get_data(location, start_day, Day(2012, 2, 2))
    assert len(mocked_data) == 32
    mocked_data = ERA5DataBase().get_data(location, start_day, Day(2013, 1, 1))
    assert len(mocked_data) == 366
    full_data = ERA5DataBase().get_data(location, Day(2010, 1, 1), Day(2015, 1, 1))
    full_data.to_csv('climate_data_daily.csv')

@pytest.mark.skip
def test_get_climate_data_for_dataset(google_earth_engine):
    google_earth_engine


@pytest.fixture()
def property_dicts():
    return [{'id': '201201', 'ou': 'Bergen', 'value': 12., 'band': 'rainfall'}, {'id': '201202', 'ou': 'Bergen', 'value': 12., 'band': 'rainfall'}, {'id': '201201', 'ou': 'Oslo', 'value': 12., 'band': 'rainfall'},          {'id': '201202', 'ou': 'Oslo', 'value': 12., 'band': 'rainfall'},
            {'id': '201201', 'ou': 'Bergen', 'value': 12., 'band': 'mean_temperature'},
             {'id': '201202', 'ou': 'Bergen', 'value': 12., 'band': 'mean_temperature'},
             {'id': '201201', 'ou': 'Oslo', 'value': 12., 'band': 'mean_temperature'},
             {'id': '201202', 'ou': 'Oslo', 'value': 12., 'band': 'mean_temperature'}]




def test_parse_properties(property_dicts):

    full_dict = parse_gee_properties(property_dicts)

    print(full_dict)


