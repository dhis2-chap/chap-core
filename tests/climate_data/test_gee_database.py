import pandas as pd

from climate_health.climate_data.gee_legacy import ERA5DataBase, parse_gee_properties
from climate_health.datatypes import Location, ClimateData, SimpleClimateData
from climate_health.google_earth_engine.gee_era5 import kelvin_to_celsium, meter_to_mm, round_two_decimal
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

"""
    Test converters
"""

def test_kelvin_to_celsium():
    assert kelvin_to_celsium(273.15) == 0
    assert kelvin_to_celsium(274.15) == 1
    assert kelvin_to_celsium(272.15) == -1

def test_meter_to_mm():
    assert meter_to_mm(1) == 1000
    assert meter_to_mm(0.1) == 100
    assert meter_to_mm(0.01) == 10

def test_round_two_decimal():
    assert round_two_decimal(-10.123) == -10.12
    assert round_two_decimal(90.1234436) == 90.12
    assert round_two_decimal(1.1234) == 1.12


@pytest.fixture()
def property_dicts():
    return [{'period': '201201', 'ou': 'Bergen', 'value': 12., 'indicator': 'rainfall'}, 
            {'period': '201202', 'ou': 'Bergen', 'value': 12., 'indicator': 'rainfall'}, 
            {'period': '201201', 'ou': 'Oslo', 'value': 12., 'indicator': 'rainfall'},          
            {'period': '201202', 'ou': 'Oslo', 'value': 12., 'indicator': 'rainfall'},
            {'period': '201201', 'ou': 'Bergen', 'value': 12., 'indicator': 'mean_temperature'},
            {'period': '201202', 'ou': 'Bergen', 'value': 12., 'indicator': 'mean_temperature'},
            {'period': '201201', 'ou': 'Oslo', 'value': 12., 'indicator': 'mean_temperature'},
            {'period': '201202', 'ou': 'Oslo', 'value': 12., 'indicator': 'mean_temperature'}]

def test_parse_properties(property_dicts):

    full_dict = parse_gee_properties(property_dicts)

    print(full_dict)


