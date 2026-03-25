from pathlib import Path
from unittest.mock import patch

import pytest

from chap_core.cli_endpoints._common import resolve_csv_path


def test_resolve_csv_path_local_path():
    path, geojson = resolve_csv_path("/some/local/file.csv")
    assert path == Path("/some/local/file.csv")
    assert geojson is None


def test_resolve_csv_path_url(tmp_path):
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("a,b\n1,2\n")

    with patch("chap_core.cli_endpoints._common.pooch.retrieve") as mock_retrieve:
        mock_retrieve.side_effect = lambda url, **kwargs: (
            str(csv_file) if ".csv" in url else (_ for _ in ()).throw(Exception("not found"))
        )
        path, geojson = resolve_csv_path("https://example.com/data.csv")

    assert path == csv_file
    assert geojson is None
    assert mock_retrieve.call_count == 2


def test_resolve_csv_path_url_with_geojson(tmp_path):
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("a,b\n1,2\n")
    geojson_file = tmp_path / "data.geojson"
    geojson_file.write_text('{"type": "FeatureCollection", "features": []}')

    def mock_retrieve(url, **kwargs):
        if url.endswith(".csv"):
            return str(csv_file)
        if url.endswith(".geojson"):
            return str(geojson_file)
        raise Exception("not found")

    with patch("chap_core.cli_endpoints._common.pooch.retrieve", side_effect=mock_retrieve):
        path, geojson = resolve_csv_path("https://example.com/data.csv")

    assert path == csv_file
    assert geojson == geojson_file
