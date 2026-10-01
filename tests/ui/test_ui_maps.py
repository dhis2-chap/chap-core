import json

from chap_core.ui.maps import BASEMAP_STYLE, NO_DATA_COLOR, bounding_box, choropleth_html, feature_location
from chap_core.ui.services import dataset_geojson, dataset_incidence, evaluation_dataset, metric_by_location


def test_dataset_incidence_covers_every_location_of_the_polygons(data_path):
    csv = data_path / "laos_subset.csv"
    values, legend = dataset_incidence(csv)
    geojson = dataset_geojson(csv)
    assert geojson is not None
    assert legend == "Annual incidence per 1000"
    assert set(values) == {feature_location(f) for f in geojson["features"]}


def test_dataset_geojson_is_none_without_polygons(data_path):
    assert dataset_geojson(data_path / "climate_data.csv") is None


def test_metric_by_location_gives_one_value_per_location(data_path):
    values = metric_by_location(data_path / "example_evaluation.nc", "crps")
    assert set(values) == {"loc1", "loc2"}


def test_evaluation_dataset_is_unknown_outside_ui_runs(data_path):
    assert evaluation_dataset(data_path / "example_evaluation.nc") is None


def test_bounding_box_spans_all_coordinates(data_path):
    geojson = dataset_geojson(data_path / "laos_subset.csv")
    assert geojson is not None
    box = bounding_box(geojson)
    assert box is not None
    west, south, east, north = box
    assert 100 < west < east < 108
    assert 13 < south < north < 23


def test_choropleth_html_colours_known_locations_over_the_basemap(data_path):
    geojson = dataset_geojson(data_path / "laos_subset.csv")
    assert geojson is not None
    first = feature_location(geojson["features"][0])
    assert first is not None
    page = choropleth_html(geojson, {first: 2.5}, "Cases & more")
    config = json.loads(page.split("const config = ", 1)[1].split(";\n", 1)[0])
    values = {feature_location(f): f["properties"].get("value") for f in config["features"]["features"]}
    assert values[first] == 2.5
    assert sum(v is None for v in values.values()) == len(values) - 1
    assert config["style"] == BASEMAP_STYLE
    assert NO_DATA_COLOR in json.dumps(config["fill"])
    assert "Cases &amp; more" in page
