import pytest
import tempfile
import json
import pathlib
import logging
from chap_core.geometry import Polygons
from chap_core.geoutils import buffer_feature, buffer_point_features, render
from chap_core.api_types import FeatureModel, FeatureCollectionModel


def test_buffer_feature_point(output_path):
    point = {'type': 'Point', 'coordinates': (10.7, 59.9)}
    distance = 0.1 # decimal degrees
    feature = FeatureModel.model_validate({'type': 'Feature', 'properties': {}, 'geometry': point})
    feature_buffer = buffer_feature(feature, distance)
    assert 'Polygon' in feature_buffer.geometry.type
    collection = FeatureCollectionModel.model_validate({'type': 'FeatureCollection', 'features': [feature_buffer] })
    polygons = Polygons(collection)
    render(polygons).save(output_path / 'test_buffer_feature_point.png')


def test_buffer_feature_points(output_path):
    point1 = {'type': 'Point', 'coordinates': (10.7, 59.9)}
    point2 = {'type': 'Point', 'coordinates': (10.4, 63.4)}
    poly = {'type': 'Polygon', 'coordinates': [[(5.3, 60.4),(5.3+0.2, 60.4),(5.3+0.2, 60.4+0.2),(5.3, 60.4+0.2),(5.3, 60.4)]]}
    geojson = {'type': 'FeatureCollection',
               'features': [
                   {'type': 'Feature', 'properties': {}, 'geometry': point1},
                   {'type': 'Feature', 'properties': {}, 'geometry': point2},
                   {'type': 'Feature', 'properties': {}, 'geometry': poly},
               ]}
    distance = 0.1 # decimal degrees
    collection = FeatureCollectionModel.model_validate(geojson)
    collection_point_buffer = buffer_point_features(collection, distance)
    assert all('Polygon' in feature.geometry.type for feature in collection_point_buffer.features)
    assert len(collection.features) == len(collection_point_buffer.features)
    polygons = Polygons(collection)
    render(polygons).save(output_path / 'test_buffer_feature_points.png')


if __name__ == '__main__':
    import pathlib
    output_path = pathlib.Path(__file__).parent / 'test_outputs'
    #test_buffer_feature_point(output_path=output_path)
    test_buffer_feature_points(output_path=output_path)
