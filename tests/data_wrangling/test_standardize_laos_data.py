from pathlib import Path
from datetime import date
import pandas as pd
import pytest

from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.datatypes import HealthData, ClimateHealthData
from chap_core.time_period import Month
from chap_core.time_period.period_range import period_range
from tests.mocks import ClimateDataBaseMock


@pytest.fixture
def geolocator():
    from geopy.geocoders import Nominatim

    return Nominatim(user_agent="MyApp")


def get_location(name, geolocator):
    return geolocator.geocode(name)


months = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

lookup = dict(zip(months, range(12)))


def get_city_name(location):
    return location.split(maxsplit=6)[-1]


def get_date(month, year):
    return date(year, lookup[month] + 1, 1)


def get_month(s):
    month_nr = lookup[s.strip().split()[0]]
    year = int(s.strip().split()[-1])
    return Month(year, month_nr)
    # s.strip().split()[0], int(s.strip().split()[-1])


def get_data(filename: Path):
    data = pd.read_csv(filename, sep="\t", header=1)
    data["period_string"] = data["periodname"]
    data["periodname"] = [
        get_date(s.strip().split()[0], int(s.strip().split()[-1]))
        for s in data["periodname"]
    ]
    data = data.sort_values(by="periodname")
    data = data.iloc[:-2]  # remove november, december 2023
    return data


def messy_standardization_function(filename: Path, geolocator):
    data = get_data(filename)
    month = [get_month(s) for s in data["period_string"]]
    print(month[0].month, month[-1].month)
    time_period = period_range(month[0], month[-1], exclusive_end=False)
    data_dict = {
        get_city_name(c): HealthData(time_period, data[c]) for c in data.columns[1:]
    }
    from chap_core._legacy_dataset import SpatioTemporalDict

    return SpatioTemporalDict(data_dict)


def link_up_geo_data(data: DataSet[HealthData], geolocator):
    climate_database = ClimateDataBaseMock()
    full_data = {}
    for city, health_data in data.items():
        location = get_location(city, geolocator)
        if location is None:
            print("Could not geocode", city)
            continue
        climate_data = climate_database.get_data(
            location, health_data.time_period[0], health_data.time_period[-1]
        )
        full_data[city] = ClimateHealthData.combine(health_data, climate_data)
    return DataSet(full_data)


@pytest.fixture
def laos_data_path(data_path):
    return data_path / "obfuscated_laos_data.tsv"


@pytest.mark.xfail
def test_standardize_laos_data(laos_data_path, geolocator):
    true_standardized = messy_standardization_function(laos_data_path, geolocator)
    print(true_standardized)
    full_data = link_up_geo_data(true_standardized, geolocator)
    print(full_data)
    schema = {}
    our_standardized = standardize_data(laos_data_path, schema)
    assert true_standardized == our_standardized
