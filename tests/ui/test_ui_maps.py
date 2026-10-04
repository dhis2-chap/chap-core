import base64
import gzip
import json

import pytest

from chap_core.ui.maps import (
    BASEMAP_STYLE,
    NO_DATA_COLOR,
    bounding_box,
    choropleth_html,
    choropleth_url,
    feature_location,
    pack_features,
)
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
    features = json.loads(gzip.decompress(base64.b64decode(config["features"])))
    values = {feature_location(f): f["properties"].get("value") for f in features["features"]}
    assert values[first] == 2.5
    assert sum(v is None for v in values.values()) == len(values) - 1
    assert config["style"] == BASEMAP_STYLE
    assert NO_DATA_COLOR in json.dumps(config["fill"])
    assert "Cases &amp; more" in page


def test_choropleth_url_carries_the_whole_map_page(data_path):
    geojson = dataset_geojson(data_path / "laos_subset.csv")
    assert geojson is not None
    url = choropleth_url(geojson, {}, "Cases")
    assert url.startswith("data:text/html;base64,")
    assert base64.b64decode(url.split(",", 1)[1]).decode() == choropleth_html(geojson, {}, "Cases")


def test_the_map_url_packs_the_regions_well_below_their_geojson_size(data_path):
    # Chrome refuses data: URLs over 2 MB; the published Laos provinces are 1.8 MB of raw GeoJSON.
    geojson = dataset_geojson(data_path / "laos_subset.csv")
    assert geojson is not None
    assert len(choropleth_url(geojson, {}, "Cases")) < len(json.dumps(geojson)) / 2


def test_the_same_map_packs_to_the_same_bytes(data_path, monkeypatch):
    import time

    geojson = dataset_geojson(data_path / "laos_subset.csv")
    assert geojson is not None
    first = pack_features(geojson)
    monkeypatch.setattr(time, "time", lambda: 4_000_000_000.0)  # a later second, as gzip's timestamp sees it
    assert pack_features(geojson) == first


def test_regions_without_a_value_are_packed_as_valid_json(data_path):
    geojson = dataset_geojson(data_path / "laos_subset.csv")
    assert geojson is not None
    first = feature_location(geojson["features"][0])
    assert first is not None
    page = choropleth_html(geojson, {first: float("nan")}, "Cases")
    config = json.loads(page.split("const config = ", 1)[1].split(";\n", 1)[0])
    # The browser parses the packed features with JSON.parse, which refuses NaN.
    features = json.loads(gzip.decompress(base64.b64decode(config["features"])), parse_constant=pytest.fail)
    assert all("value" not in f["properties"] for f in features["features"])


def test_region_names_cannot_close_the_maps_script():
    from chap_core.ui.maps import choropleth_html

    geojson = {"type": "FeatureCollection", "features": [{"type": "Feature", "id": "a",
               "properties": {"name": "</script><script>alert(1)</script>"},
               "geometry": {"type": "Point", "coordinates": [0, 0]}}]}  # fmt: skip
    page = choropleth_html(geojson, {"a": 1.0}, "Cases")
    assert "</script><script>alert(1)" not in page
