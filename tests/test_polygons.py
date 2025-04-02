import pytest
import tempfile
import json
import pathlib
import logging
from chap_core.geometry import Polygons
from chap_core.geoutils import simplify_topology, render


def test_to_from_geojson_file(data_path):
    polygons = Polygons.from_file(data_path / "example_polygons.geojson")

    with tempfile.NamedTemporaryFile() as f:
        polygons.to_file(f.name)
        polygons2 = Polygons.from_file(f.name)

        assert polygons2 == polygons


def test_simplify_polygons(data_path, output_path):
    # load data
    print('loading')
    polygons = Polygons.from_file(data_path / "small_laos_data_with_polygons.geojson")
    render(polygons).save(output_path / 'test_simplify_polygons - before.png')
    
    # simplify data
    print('simplifying')
    simplified = simplify_topology(polygons)
    render(simplified).save(output_path / 'test_simplify_polygons - after.png')

    # simplify must return same number of features
    count1 = len(polygons)
    count2 = len(simplified)
    assert count1 == count2

    # simplify must return smaller json data size
    print('geo interface')
    size1 = len(json.dumps(polygons.__geo_interface__))
    size2 = len(json.dumps(simplified.__geo_interface__))
    print(f'total geojson byte size reduced from {size1/1000}kb to {size2/1000}kb')
    assert size2 < size1


def test_laos_polygons(data_path):
    polygons = Polygons.from_file(data_path / "small_laos_data_with_polygons.geojson").data
    #polygons = Polygons.from_file(data_path / "small_laos_data_with_polygons.geojson", id_property='district').data


if __name__ == '__main__':
    import pathlib
    data_path = pathlib.Path(__file__).parent.parent / 'example_data'
    test_simplify_polygons(data_path)
