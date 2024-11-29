import os
from datetime import datetime, timezone

from dotenv import find_dotenv, load_dotenv
import pandas as pd

from chap_core.api_types import FeatureCollectionModel
from chap_core.datatypes import GEEData, HealthPopulationData, tsdataclass
from chap_core.google_earth_engine.gee_era5 import (
    Band,
    Era5LandGoogleEarthEngine,
    kelvin_to_celsium,
    meter_to_mm,
)
from chap_core.google_earth_engine.gee_era5 import (
    Era5LandGoogleEarthEngineHelperFunctions,
)
from chap_core.google_earth_engine.gee_raw import fetch_era5_data, GEECredentials
from chap_core.google_earth_engine.multi_resolution import harmonize_with_daily_data, pack_daily_data
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.time_period.date_util_wrapper import Month, PeriodRange
import pytest
import ee as _ee

era5_land_gee_helper = Era5LandGoogleEarthEngineHelperFunctions()


@pytest.fixture()
def ee(era5_land_gee):
    return _ee


@pytest.fixture()
def era5_land_gee():
    t = Era5LandGoogleEarthEngine()
    if not t.is_initialized:
        pytest.skip("Google Earth Engine not available")
    return t


def test_kelvin_to_celsium():
    assert kelvin_to_celsium(272.15) == -1


def test_meter_to_mm():
    assert meter_to_mm(0.01) == 10


def test_round_two_decimal():
    assert round(1.1234, 2) == 1.12


"""
    Test parse_properties
"""


@pytest.fixture()
def property_dicts():
    return [
        {"period": "201201", "ou": "Bergen", "value": 12.0, "indicator": "rainfall"},
        {"period": "201202", "ou": "Bergen", "value": 12.0, "indicator": "rainfall"},
        {"period": "201201", "ou": "Oslo", "value": 12.0, "indicator": "rainfall"},
        {"period": "201202", "ou": "Oslo", "value": 12.0, "indicator": "rainfall"},
        {
            "period": "201201",
            "ou": "Bergen",
            "value": 12.0,
            "indicator": "mean_temperature",
        },
        {
            "period": "201202",
            "ou": "Bergen",
            "value": 12.0,
            "indicator": "mean_temperature",
        },
        {
            "period": "201201",
            "ou": "Oslo",
            "value": 12.0,
            "indicator": "mean_temperature",
        },
        {
            "period": "201202",
            "ou": "Oslo",
            "value": 12.0,
            "indicator": "mean_temperature",
        },
    ]


def test_parse_gee_properties(property_dicts):
    result: DataSet = era5_land_gee_helper.parse_gee_properties(property_dicts)
    assert result is not None
    assert len(result.to_pandas()) == 4
    assert (result.get_location("Oslo").data().mean_temperature == [12, 12]).all()


"""
    Test get_image_for_periode, tests for multiple bands and periodes
"""


@pytest.fixture()
def collection(ee):
    return ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")


@pytest.fixture(
    params=[
        Band(
            name="temperature_2m",
            reducer="mean",
            periode_reducer="mean",
            converter=kelvin_to_celsium,
            indicator="mean_temperature",
        ),
        Band(
            name="total_precipitation_sum",
            reducer="mean",
            periode_reducer="sum",
            converter=meter_to_mm,
            indicator="rainfall",
        ),
    ]
)
def band(request):
    return request.param


@pytest.fixture()
def periode(ee):
    return ee.Dictionary(
        {"period": "1", "start_date": "2023-01-01", "end_date": "2023-01-02"}
    )


def test_get_period(band: Band, collection, periode):
    image: ee.Image = era5_land_gee_helper.get_image_for_period(
        periode, band, collection
    )

    fetched_image = image.getInfo()

    assert fetched_image is not None
    assert fetched_image["type"] == "Image"
    assert len(fetched_image["bands"]) == 1
    assert fetched_image["bands"][0]["id"] == band.name
    assert fetched_image["properties"]["system:time_start"] == int(
        (
            datetime.strptime(periode.getInfo().get("start_date"), "%Y-%m-%d")
            .replace(tzinfo=timezone.utc)
            .timestamp()
        )
        * 1000
    )
    assert fetched_image["properties"]["system:time_end"] == int(
        (
            datetime.strptime(periode.getInfo().get("end_date"), "%Y-%m-%d")
            .replace(tzinfo=timezone.utc)
            .timestamp()
        )
        * 1000
    )


"""
    Test create_ee_dict
"""


@pytest.fixture()
def time_periode(ee):
    return Month(2023, 1)


def test_create_ee_dict(time_periode):
    # NotImplementedError: Must be implemented in subclass
    dict = era5_land_gee_helper.create_ee_dict(time_periode)
    assert dict is not None
    # assert dict.get("period") == "202301"


"""
    Test create_ee_feature
"""


@pytest.fixture()
def ee_feature(ee):
    return ee.Feature(
        ee.Geometry.Point([-114.318, 38.985]), {"system:index": "abc123", "mean": 244}
    )


@pytest.fixture()
def ee_image(ee):
    image = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").first()
    return image.set(
        {
            "system:indicator": "temperature_2m",
            "system:period": "2014-03",
        }
    )


def test_creat_ee_feature(ee_feature, ee_image):
    feature = era5_land_gee_helper.creat_ee_feature(
        ee_feature, ee_image, "mean"
    ).getInfo()

    assert feature is not None
    assert feature["properties"]["ou"] == "abc123"
    assert feature["properties"]["value"] == 244
    assert feature["geometry"] == None
    assert feature["properties"]["indicator"] == "temperature_2m"
    assert feature["properties"]["period"] == "2014-03"


"""
    Test convert_value_by_band_converter
"""


@pytest.fixture()
def list_of_bands():
    return [
        Band(
            name="temperature_2m",
            reducer="mean",
            periode_reducer="mean",
            converter=kelvin_to_celsium,
            indicator="mean_temperature",
        ),
        Band(
            name="total_precipitation_sum",
            reducer="mean",
            periode_reducer="sum",
            converter=meter_to_mm,
            indicator="rainfall",
        ),
    ]


@pytest.fixture()
def data():
    return [
        {"properties": {"v1": "100", "indicator": "mean_temperature", "value": 300}},
        {"properties": {"v1": "200", "indicator": "rainfall", "value": 0.004}},
    ]


def test_convert_value_by_band_converter(data, list_of_bands):
    result = era5_land_gee_helper.convert_value_by_band_converter(data, list_of_bands)

    assert result is not None
    assert len(result) == 2

    # test converters
    assert result[0]["value"] == 26.85
    assert result[1]["value"] == 4

    # other properties
    assert result[0]["indicator"] == "mean_temperature"
    assert result[1]["indicator"] == "rainfall"
    assert result[0]["v1"] == "100"
    assert result[1]["v1"] == "200"


@pytest.fixture()
def feature_collection(ee):
    geojson = {
        "type": "FeatureCollection",
        "columns": {},
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [0, 0]},
                "id": "1_2_0_fdc6uOvgoji",
                "properties": {
                    "indicator": "mean_temperature",
                    "ou": "fdc6uOvgoji",
                    "period": "202201",
                    "value": 301.6398539038109,
                },
            },
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [0, 0]},
                "id": "2_11_fdc6uOvgoji",
                "properties": {
                    "indicator": "rainfall",
                    "ou": "fdc6uOvgoji",
                    "period": "202212",
                    "value": 0.01885525397859519,
                },
            },
        ],
    }

    return ee.FeatureCollection(geojson)


def test_value_collection_to_list(feature_collection):
    result = era5_land_gee_helper.feature_collection_to_list(feature_collection)

    assert result is not None
    assert len(result) == 2

    assert result[0]["properties"]["indicator"] == "mean_temperature"
    assert result[1]["properties"]["indicator"] == "rainfall"
    assert result[0]["properties"]["value"] == 301.6398539038109
    assert result[1]["properties"]["value"] == 0.01885525397859519


@pytest.fixture()
def gee_credentials():
    try:
        load_dotenv(find_dotenv())
        account = os.environ.get("GOOGLE_SERVICE_ACCOUNT_EMAIL")
        private_key = os.environ.get("GOOGLE_SERVICE_ACCOUNT_PRIVATE_KEY")
        return GEECredentials(account=account, private_key=private_key)
    except Exception:
        pytest.skip("Google Earth Engine not available")


@pytest.fixture()
def polygons(polygon_json):
    return FeatureCollectionModel.model_validate_json(polygon_json)


@pytest.fixture()
def polygon_json(data_path):
    return open(data_path / "Organisation units.geojson").read()

def test_get_daily_data(era5_land_gee, polygons, data_path):
    polygons.features = polygons.features[:2]
    period_range = [
            Month(2023, 1),
            Month(2023, 2),
        ]
    period_range = PeriodRange.from_period_list(False, period_range)
    
    data = era5_land_gee.get_daily_data(
        polygons.model_dump(),
        period_range,
    )
    assert data is not None
    print(data)
    assert len(data) > 2 *28* len(polygons.features)
    data.to_csv(data_path / "era5_land_daily_data.csv", index=False)


def test_pack_daily_data(data_path, tmp_path):
    data = pd.read_csv(data_path / "era5_land_daily_data.csv")
    period_range = [Month(2023, 1), Month(2023, 2)]
    period_range = PeriodRange.from_period_list(False, period_range)
    data = pack_daily_data(data, period_range, GEEData)
    assert len(data.locations()) == 2
    for location, d in data.items():
        assert len(d.time_period) == 2
        assert d.temperature_2m.shape == (2, 31)
        assert d.total_precipitation_sum.shape == (2, 31)

    data.to_pickle(tmp_path / "era5_land_daily_data.pkl")
    new_data = DataSet.from_pickle(tmp_path / "era5_land_daily_data.pkl", GEEData)
    assert len(new_data.locations()) == 2
    for location, d in new_data.items():
        assert len(d.time_period) == 2
        assert d.temperature_2m.shape == (2, 31), d.temperature_2m
        assert d.total_precipitation_sum.shape == (2, 31), d.total_precipitation_sum

def test_harmonize_daily_data(polygons):
    polygons.features = polygons.features[:2]
    data = HealthPopulationData(PeriodRange.from_period_list(False, [Month(2023, 1), Month(2023, 2)]),
                                       disease_cases=[1, 2], population=[100, 200])
    health_population_data = DataSet({f.id: data for f in polygons.features})
    health_population_data.set_polygons(polygons)
    
    @tsdataclass
    class NewClass(HealthPopulationData):
        temperature_2m: float
        total_precipitation_sum: float

    dataset = harmonize_with_daily_data(health_population_data, GEEData, NewClass)
    assert set(dataset.keys()) == set(health_population_data.keys())
    for location, d in dataset.items():
        assert len(d.time_period) == 2
        assert d.temperature_2m.shape == (2, 31)
        assert d.total_precipitation_sum.shape == (2, 31)
        assert d.disease_cases.shape == (2,)
        assert d.population.shape == (2,)


@pytest.mark.skip("Calling actual gee data")
def test_gee_api(gee_credentials, polygons):
    data = fetch_era5_data(
        gee_credentials,
        polygons,
        start_period="202201",
        end_period="202202",
        band_names=["temperature_2m", "total_precipitation_sum"],
    )
    print(data)
    assert len(data) == 2 * 2 * len(polygons.features)


@pytest.mark.skip("Calling actual gee data")
def test_gee_api_simple(gee_credentials, polygon_json):
    data = fetch_era5_data(
        gee_credentials.model_dump(),
        polygon_json,
        start_period="202201",
        end_period="202202",
        band_names=["temperature_2m", "total_precipitation_sum"],
    )
    print(data)
