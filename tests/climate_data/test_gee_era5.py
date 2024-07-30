"""
    Test converters
"""

from datetime import datetime, timezone
from typing import List

from dotenv import find_dotenv, load_dotenv
from climate_health.google_earth_engine.gee_era5 import Band, Era5LandGoogleEarthEngine, Periode, kelvin_to_celsium, \
    meter_to_mm, round_two_decimal
from climate_health.google_earth_engine.gee_era5 import SpatioTemporalDictConverter, \
    Era5LandGoogleEarthEngineHelperFunctions
from climate_health.time_period.date_util_wrapper import TimePeriod
import pytest
import ee as _ee

spatio_temporal_dict_converter = SpatioTemporalDictConverter()
era5_land_gee_helper = Era5LandGoogleEarthEngineHelperFunctions()

# need to

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
    spatio_temporal_dict_converter.parse_gee_properties(property_dicts)


def test_get_feature_from_zip(tests_path):
    features = era5_land_gee_helper.get_feature_from_zip(tests_path/"climate_data/fixtures/test_chapdata-bombali-jan2022-dec2022.zip")
    assert features is not None


"""
    Test get_image_for_periode
"""


@pytest.fixture()
def collection():
    return ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')


@pytest.fixture(
    params=[
        Band(name="temperature_2m", reducer="mean", periodeReducer="mean", converter=kelvin_to_celsium,
             indicator="mean_temperature"),
        Band(name="total_precipitation_sum", reducer="mean", periodeReducer="sum", converter=meter_to_mm,
             indicator="rainfall")
    ]
)
def band(request):
    return request.param


def get_ee_params():
    try:
        return [ee.Dictionary({"period": "1", "start_date": "2023-01-01", "end_date": "2023-01-02"}),
                ee.Dictionary({"period": "2", "start_date": "2021-01-31", "end_date": "2022-01-01"}),
                ee.Dictionary({"period": "3", "start_date": "1970-01-01", "end_date": "1971-01-02"})]
    except Exception as e:

        return []


@pytest.fixture(
    params=get_ee_params()
)
def periode(request):
    return request.param


def test_get_periode(band: Band, collection, periode):
    image: ee.Image = era5_land_gee_helper.get_image_for_periode(periode, band, collection)

    fetched_image = image.getInfo()

    assert fetched_image is not None
    assert fetched_image["type"] == "Image"
    assert len(fetched_image['bands']) == 1
    assert fetched_image['bands'][0]['id'] == band.name
    assert fetched_image['properties']['system:time_start'] == int((datetime.strptime(
        periode.getInfo().get("start_date"), "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()) * 1000)
    assert fetched_image['properties']['system:time_end'] == int((datetime.strptime(periode.getInfo().get("end_date"),
                                                                                    "%Y-%m-%d").replace(
        tzinfo=timezone.utc).timestamp()) * 1000)


"""
    Test create_ee_dict 
"""


@pytest.fixture()
def time_periode():
    # Needs to create a TimePeriod here
    return TimePeriod(2023, 1)


@pytest.mark.skip(reason="Return: Must be implemented in subclass")
def test_create_ee_dict(time_periode):
    # NotImplementedError: Must be implemented in subclass
    era5_land_gee_helper.create_ee_dict(time_periode)
    pass


"""
    Test create_ee_feature
"""


@pytest.fixture()
def ee_feature(ee):
    return ee.Feature(ee.Geometry.Point([-114.318, 38.985]), {'system:index': 'abc123', 'mean': 244})


@pytest.fixture()
def ee_image(ee):
    image = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").first()
    return image.set({
        "system:indicator": "temperature_2m",
        "system:period": "2014-03",
    })


def test_creat_ee_feature(ee_feature, ee_image):
    feature = era5_land_gee_helper.creat_ee_feature(ee_feature, ee_image, "mean").getInfo()

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
        Band(name="temperature_2m", reducer="mean", periodeReducer="mean", converter=kelvin_to_celsium,
             indicator="mean_temperature"),
        Band(name="total_precipitation_sum", reducer="mean", periodeReducer="sum", converter=meter_to_mm,
             indicator="rainfall")
    ]


@pytest.fixture()
def data():
    return [
        {"properties": {"v1": "100", "v2": "200", "indicator": "mean_temperature", "value": 400}},
    ]


def test_convert_value_by_band_converter(data, list_of_bands):
    result = era5_land_gee_helper.convert_value_by_band_converter(data, list_of_bands)

    assert result is not None

    # print(result)


def value_collection_to_list():
    pass
    # era5_land_gee_helper.value_collection_to_list
