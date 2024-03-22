from climate_health.climate_data.gee import ERA5DataBase
from climate_health.datatypes import Location, ClimateData
from climate_health.time_period import Month, Day, TimePeriod
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


def test_get_data(mocker):
    mocker.patch('climate_health.climate_data.gee.ee.Initialize')
    mocker.patch('climate_health.climate_data.gee.ee.Authenticate')
    mock_get_image_collection = mocker.patch('climate_health.climate_data.gee.get_image_collection')
    mock_ic = mocker.MagicMock()
    mock_get_image_collection.return_value = mock_ic

    mock_point = mocker.MagicMock()
    mocker.patch('climate_health.climate_data.gee.ee.Geometry.Point', return_value=mock_point)

    mock_feature_collection = mocker.MagicMock()
    mock_ic.filterDate.return_value = mock_ic
    mock_ic.select.return_value = mock_ic
    mock_ic.map.return_value = mock_feature_collection

    mock_feature_collection.getInfo.return_value = {
        'features': [
            {'properties': {'temperature_2m': 290.15, 'temperature_2m_max': 295.15, 'total_precipitation_sum': 1.0},
             'id': '20200101'}
        ]
    }

    database = ERA5DataBase()

    test_region = Location(longitude=123.45, latitude=54.321)
    test_start = TimePeriod.parse("2020-03-15")
    test_end = TimePeriod.parse("2021-03-15")

    result = database.get_data(test_region, test_start, test_end)

    assert isinstance(result, ClimateData)
