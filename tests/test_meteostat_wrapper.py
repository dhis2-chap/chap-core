from datetime import datetime

import pandas as pd
import pytest
from meteostat import Point

from chap_core.climate_data.meteostat_wrapper import ClimateDataMeteoStat


def test_format_start_date():
    list_start_date = [
        "2010-01-01",  # Daily
        "2010-W1",  # Weekly
        "2010-1",  # Monthly
        "2010",  # Yearly
    ]

    res_date = []

    climate_lookup = ClimateDataMeteoStat()
    for start_date in list_start_date:
        res_date.append(climate_lookup._format_start_date(start_date))

    assert res_date == [
        datetime(2010, 1, 1, 0, 0),
        datetime(2010, 1, 4, 0, 0),
        datetime(2010, 1, 1, 0, 0),
        datetime(2010, 1, 1, 0, 0),
    ]


def test_format_end_date():
    list_start_date = [
        "2010-01-01",  # Daily
        "2010-W1",  # Weekly
        "2010-1",  # Monthly
        "2010",  # Yearly
    ]

    res_date = []

    climate_lookup = ClimateDataMeteoStat()
    for start_date in list_start_date:
        res_date.append(climate_lookup._format_end_date(start_date))

    assert res_date == [
        datetime(2010, 1, 1, 0, 0),
        datetime(2010, 1, 10, 0, 0),
        datetime(2010, 1, 1, 0, 0),
        datetime(2010, 1, 1, 0, 0),
    ]


@pytest.mark.xfail(reason="Do not work with CI")
def test_fetch_location():
    location = "Paris"

    climate_lookup = ClimateDataMeteoStat()
    point_location = climate_lookup._fetch_location(location)

    point_expected = Point(48.8535, 2.3484, 0)

    assert point_location._lat == point_expected._lat
    assert point_location._lon == point_expected._lon


@pytest.mark.xfail(reason="Do not work with CI")
def test_get_climate_daily():
    location = "Paris"
    start_date = "2010-01-01"
    end_date = "2010-01-03"

    climate_dataframe = ClimateDataMeteoStat().get_climate(
        location, start_date, end_date
    )

    data = pd.DataFrame(
        {
            "time_period": ["2010-1-1", "2010-1-2", "2010-1-3"],
            "rainfall": [0.3, 0.0, 0.0],
            "mean_temperature": [1.4, 1.4, 0.8],
            "max_temperature": [3.0, 4.0, 5.2],
        }
    )

    assert climate_dataframe.equals(data)


@pytest.mark.xfail(reason="Do not work with CI")
def test_get_climate_weekly():
    location = "Paris"
    start_date = "2010-W1"
    end_date = "2010-W3"

    climate_dataframe = ClimateDataMeteoStat().get_climate(
        location, start_date, end_date
    )

    data = pd.DataFrame(
        {
            "time_period": ["2010-W1", "2010-W2", "2010-W3"],
            "rainfall": [1.3, 11.5, 9.0],
            "mean_temperature": [-2.0, 3.4, 4.7],
            "max_temperature": [1.2, 10.4, 9.4],
        }
    )

    assert climate_dataframe.equals(data)


@pytest.mark.xfail(reason="Do not work with CI")
def test_get_climate_monthly():
    location = "Paris"
    start_date = "2010-01"
    end_date = "2010-03"

    climate_dataframe = ClimateDataMeteoStat().get_climate(
        location, start_date, end_date
    )

    data = pd.DataFrame(
        {
            "time_period": ["2010-1", "2010-2", "2010-3"],
            "rainfall": [26.0, 48.0, 40.0],
            "mean_temperature": [2.0, 4.9, 8.6],
            "max_temperature": [3.8, 8.0, 12.8],
        }
    )

    assert climate_dataframe.equals(data)


@pytest.mark.xfail(reason="Do not work with CI")
def test_get_climate_yearly():
    location = "Paris"
    start_date = "2010"
    end_date = "2012"

    climate_dataframe = ClimateDataMeteoStat().get_climate(
        location, start_date, end_date
    )

    data = pd.DataFrame(
        {
            "time_period": ["2010", "2011", "2012"],
            "rainfall": [596.2, 543.3, 25.2],
            "mean_temperature": [11.9, 13.8, 7.6],
            "max_temperature": [27.3, 25.3, 10.1],
        }
    )

    assert climate_dataframe.equals(data)


def test_make_date_range_daily():
    start_date = "2010-01-01"
    end_date = "2010-01-03"

    climate_lookup = ClimateDataMeteoStat()
    climate_lookup._delta = "day"
    date_range = climate_lookup._make_date_range(start_date, end_date)

    assert date_range == ["2010-1-1", "2010-1-2", "2010-1-3"]


def test_make_date_range_weekly():
    start_date = "2010-W1"
    end_date = "2010-W3"

    climate_lookup = ClimateDataMeteoStat()
    climate_lookup._delta = "week"
    date_range = climate_lookup._make_date_range(start_date, end_date)

    assert date_range == ["2010-W1", "2010-W2", "2010-W3"]


def test_make_date_range_monthly():
    start_date = "2010-1"
    end_date = "2010-3"

    climate_lookup = ClimateDataMeteoStat()
    climate_lookup._delta = "month"
    date_range = climate_lookup._make_date_range(start_date, end_date)

    assert date_range == ["2010-1", "2010-2", "2010-3"]


def test_make_test_range_yearly():
    start_date = "2010"
    end_date = "2012"

    climate_lookup = ClimateDataMeteoStat()
    climate_lookup._delta = "year"
    date_range = climate_lookup._make_date_range(start_date, end_date)

    assert date_range == ["2010", "2011", "2012"]


def test_fetch_climate_data():
    location = Point(48.8534, 2.3488, 0)
    start_date = datetime(2010, 1, 1, 0, 0)
    end_date = datetime(2010, 1, 3, 0, 0)

    climate_data = ClimateDataMeteoStat()
    climate_data._delta = "day"
    climate_dataframe = climate_data._fetch_climate_data(location, start_date, end_date)

    print(climate_dataframe)
    data = pd.DataFrame(
        {
            "rainfall": [0.3, 0.0, 0.0],
            "mean_temperature": [1.4, 1.4, 0.8],
            "max_temperature": [3.0, 4.0, 5.2],
        }
    )

    data = data.set_index(pd.date_range(start_date, end_date, freq="D"))

    assert climate_dataframe.equals(data)
