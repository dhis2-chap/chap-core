import io
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from shapely.geometry import GeometryCollection, Point

from chap_core.file_io import dataset_files
from chap_core.file_io.dataset_files import (
    ColumnInfo,
    DatasetFileError,
    DatasetFileMetadata,
    DatasetInfo,
    read_locations,
    read_observations,
    write_dataset_files,
)
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet


def _parquet_bytes(frame: pd.DataFrame, extra_metadata: dict[bytes, bytes] | None = None) -> bytes:
    table = pa.Table.from_pandas(frame, preserve_index=False)
    table = table.replace_schema_metadata({**(table.schema.metadata or {}), **(extra_metadata or {})})
    buffer = io.BytesIO()
    pq.write_table(table, buffer)
    return buffer.getvalue()


def test_dataset_files_round_trip(data_path, tmp_path):
    dataset = DataSet.from_csv(data_path / "laos_subset.csv")
    metadata = DatasetFileMetadata(
        dataset=DatasetInfo(name="Laos", target_column="disease_cases"),
        columns={"rainfall": ColumnInfo(unit="mm", dhis2_data_element="de_rain")},
        provenance={"license": "CC-BY-4.0"},
    )
    observations_path, locations_path = write_dataset_files(dataset, tmp_path, metadata)

    frame, read_back = read_observations(observations_path.read_bytes())
    assert read_back == metadata
    assert set(frame.columns) == {"location", "time_period", *dataset.field_names()}
    for location, data in dataset.items():
        rows = frame[frame["location"] == location]
        assert rows["time_period"].tolist() == [period.id for period in data.time_period]
        for name in dataset.field_names():
            np.testing.assert_allclose(rows[name].to_numpy(), getattr(data, name), equal_nan=True)

    locations = read_locations(locations_path.read_bytes())
    original = {feature.id: feature for feature in dataset.polygons.features}
    assert {feature.id for feature in locations.features} == set(original)
    for feature in locations.features:
        source = original[feature.id]
        assert feature.properties is not None and source.properties is not None
        assert feature.properties["parent"] == source.properties["parent"]
        assert feature.geometry is not None and source.geometry is not None
        assert feature.geometry.type == source.geometry.type


def test_read_observations_reads_nulls_as_missing(observations_frame):
    read, metadata = read_observations(_parquet_bytes(observations_frame))
    assert np.isnan(read["disease_cases"].iloc[1])
    assert metadata == DatasetFileMetadata()


def test_read_observations_ignores_other_binary_metadata(observations_frame):
    read, _ = read_observations(_parquet_bytes(observations_frame, {b"vendor": b"\xff"}))
    assert len(read) == 2


def test_read_locations_accepts_geojson(data_path):
    locations = read_locations((data_path / "laos_subset.geojson").read_bytes())
    assert len(locations.features) == 3


@pytest.mark.parametrize(
    ("change", "metadata", "message"),
    [
        (lambda frame: frame.drop(columns="time_period"), None, "no time_period column"),
        (lambda frame: frame.assign(parent="p"), None, "not numeric: parent"),
        (None, {b"chap.spec_version": b"2"}, "this Chap reads version '1'"),
        (None, {b"chap.dataset": json.dumps({"masked_ranges": "all"}).encode()}, "Invalid chap.\\* metadata"),
        (None, {b"chap.dataset": b"\xff"}, "chap.dataset is not UTF-8 text"),
        (None, {b"chap.provenance": json.dumps({"note": "x" * 70_000}).encode()}, "chap.provenance is larger"),
    ],
)
def test_read_observations_rejects(observations_frame, change, metadata, message):
    frame = change(observations_frame) if change else observations_frame
    with pytest.raises(DatasetFileError, match=message):
        read_observations(_parquet_bytes(frame, metadata))


@pytest.mark.parametrize(
    ("limit", "value", "message"), [("MAX_ROWS", 1, "has 2 rows"), ("MAX_DECODED_BYTES", 10, "decodes to")]
)
def test_read_observations_checks_the_footer_before_decoding(observations_frame, monkeypatch, limit, value, message):
    monkeypatch.setattr(dataset_files, limit, value)
    with pytest.raises(DatasetFileError, match=message):
        read_observations(_parquet_bytes(observations_frame))


def test_read_locations_rejects_parquet_without_geometry(observations_frame):
    with pytest.raises(DatasetFileError, match="not GeoParquet"):
        read_locations(_parquet_bytes(observations_frame))


def test_read_locations_rejects_geometry_chap_cannot_use(observations_frame):
    locations = gpd.GeoDataFrame(
        observations_frame[["location"]].drop_duplicates(), geometry=[GeometryCollection([Point(0, 0)])], crs=4326
    )
    buffer = io.BytesIO()
    locations.to_parquet(buffer, index=False)
    with pytest.raises(DatasetFileError, match="geometry Chap cannot use"):
        read_locations(buffer.getvalue())
