import os

import pytest
from chap_core.geojson import geojson_to_shape, geojson_to_graph
from . import EXAMPLE_DATA_PATH


@pytest.fixture
def geojson_example_file():
    return EXAMPLE_DATA_PATH / "Organisation units.geojson"


@pytest.mark.skip
def test_geojson_to_shape(geojson_example_file):
    # this does not work with temporarifles, as a real directory is needed
    out = "shapefile_test"
    geojson_to_shape(geojson_example_file, out + ".shp")

    extensions = [".cpg", ".dbf", ".prj", ".shp", ".shx"]
    for extension in extensions:
        assert os.path.isfile(out + extension)
        os.remove(out + extension)


def test_geojson_to_graph(geojson_example_file, tmp_path):
    filename = tmp_path / "graph.txt"
    res = geojson_to_graph(geojson_example_file, filename)
    print(res)
