import pytest
from chap_core.geojson import geojson_to_graph
from . import EXAMPLE_DATA_PATH


@pytest.fixture
def geojson_example_file():
    return EXAMPLE_DATA_PATH / "Organisation units.geojson"


def test_geojson_to_graph(geojson_example_file, tmp_path):
    filename = tmp_path / "graph.txt"
    geojson_to_graph(geojson_example_file, filename)
    assert filename.read_text()
