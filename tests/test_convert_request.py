import json

import pandas as pd

from chap_core.cli_endpoints.convert import convert_request


def test_convert_request_from_example_json(tmp_path):
    """Test convert_request using the example JSON fixture."""
    input_path = "example_data/create-backtest-with-data.json"
    output_prefix = tmp_path / "output"

    convert_request(input_path, output_prefix)

    csv_path = tmp_path / "output.csv"
    geojson_path = tmp_path / "output.geojson"

    assert csv_path.exists()
    assert geojson_path.exists()

    df = pd.read_csv(csv_path)
    assert "location" in df.columns
    assert "time_period" in df.columns
    assert "disease_cases" in df.columns
    assert "rainfall" in df.columns
    assert "mean_temperature" in df.columns
    assert len(df) > 0

    with open(geojson_path) as f:
        geojson = json.load(f)
    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) > 0


def test_convert_request_pivots_correctly(tmp_path):
    """Test that provided data is correctly pivoted into columns."""
    request = {
        "providedData": [
            {"featureName": "rainfall", "orgUnit": "loc_1", "period": "2022-01", "value": 1.5},
            {"featureName": "rainfall", "orgUnit": "loc_1", "period": "2022-02", "value": 2.0},
            {"featureName": "disease_cases", "orgUnit": "loc_1", "period": "2022-01", "value": 10},
            {"featureName": "disease_cases", "orgUnit": "loc_1", "period": "2022-02", "value": 15},
            {"featureName": "rainfall", "orgUnit": "loc_2", "period": "2022-01", "value": 3.0},
            {"featureName": "disease_cases", "orgUnit": "loc_2", "period": "2022-01", "value": 20},
        ],
        "geojson": {
            "type": "FeatureCollection",
            "features": [
                {
                    "id": "loc_1",
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [0, 0]},
                    "properties": {},
                },
                {
                    "id": "loc_2",
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [1, 1]},
                    "properties": {},
                },
            ],
        },
    }

    input_path = tmp_path / "request.json"
    with open(input_path, "w") as f:
        json.dump(request, f)

    output_prefix = tmp_path / "result"
    convert_request(input_path, output_prefix)

    df = pd.read_csv(tmp_path / "result.csv")
    assert set(df.columns) == {"location", "time_period", "disease_cases", "rainfall"}
    assert len(df) == 3  # 2 periods for loc_1 + 1 period for loc_2

    row = df[(df["location"] == "loc_1") & (df["time_period"] == "2022-01")].iloc[0]
    assert row["rainfall"] == 1.5
    assert row["disease_cases"] == 10

    with open(tmp_path / "result.geojson") as f:
        geojson = json.load(f)
    assert len(geojson["features"]) == 2
