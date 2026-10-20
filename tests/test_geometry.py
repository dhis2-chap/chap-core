import logging

import pytest


logging.basicConfig(level=logging.INFO)


import pytest
from geojson_pydantic import Point

from chap_core.api_types import FeatureCollectionModel, FeatureModel
from chap_core.geometry import Polygons


@pytest.fixture
def vietnam_polygons(data_path) -> Polygons:
    return Polygons.from_file(data_path / "vietnam_monthly.geojson")


def test_id_to_name_tuple_dict(vietnam_polygons: Polygons):
    vietnam_polygons.id_to_name_tuple_dict()
    for feature in vietnam_polygons.feature_collection().features:
        assert feature.id in vietnam_polygons.id_to_name_tuple_dict()
        # assert feature.properties['name'] in vietnam_polygons.id_to_name_tuple_dict().values()


def test_get_parent_dict(vietnam_polygons: Polygons):
    parent_dict = vietnam_polygons.get_parent_dict()
    for feature in vietnam_polygons.feature_collection().features:
        assert feature.id in parent_dict
        assert isinstance(parent_dict[feature.id], str)
        # assert feature.properties['parent'] in parent_dict.values()
        # assert feature.properties['parent'] in parent_dict.values()


def test_geojson_round_trip():
    feature = FeatureModel(
        type="Feature",
        id="loc_1",
        properties={"name": "Location 1"},
        geometry=Point(type="Point", coordinates=[10.0, 20.0]),
    )
    collection = FeatureCollectionModel(
        type="FeatureCollection",
        features=[feature],
    )

    json_str = collection.model_dump_json()
    restored = FeatureCollectionModel.model_validate_json(json_str)

    assert len(restored.features) == 1
    assert restored.features[0].id == "loc_1"
    assert restored.features[0].properties is not None
    assert restored.features[0].properties["name"] == "Location 1"
    assert restored.features[0].geometry is not None
    assert restored.features[0].geometry.type == "Point"
