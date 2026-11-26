"""Read and write a dataset as two Parquet files: observations and locations.

``observations.parquet`` is wide: a ``location`` and a ``time_period`` column, then one numeric
column per feature. ``locations.parquet`` is GeoParquet with one row per location: ``location``,
optional ``name``, ``parent`` and ``level``, and the geometry. GeoJSON is also accepted for the
locations.

Dataset settings travel in the files' own Parquet key-value metadata, under ``chap.*`` keys.
pandas drops that metadata, so it is always read from and written to the Arrow schema directly.
"""

import io
import json
import math
from pathlib import Path
from typing import Any, cast

import geopandas as gpd
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, Field
from pydantic import ValidationError as PydanticValidationError
from shapely.geometry import shape

from chap_core.api_types import FeatureCollectionModel
from chap_core.database.dataset_tables import MaskedRange
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

SPEC_VERSION = "1"
PARQUET_MAGIC = b"PAR1"
MAX_FILE_BYTES = 100 * 1024 * 1024
# Two files, each of which a DHIS2 route may send twice (DHIS2-22596), plus form overhead.
MAX_UPLOAD_BYTES = 4 * MAX_FILE_BYTES + 1024 * 1024
# Limits checked in the Parquet footer before any data is decoded, so a small, highly compressed
# file cannot expand into more memory than the API process has. Phase 1 imports through the same
# Python lists as make-dataset, about 550 bytes per row at the peak (400 MB measured for 730,000
# rows), so 1 million rows keeps one upload near 550 MB. Larger datasets wait for stored files.
MAX_ROWS = 1_000_000
MAX_DECODED_BYTES = 512 * 1024 * 1024
MAX_METADATA_BYTES = 64 * 1024
# Row groups of this size let readers skip to one org unit without reading the whole file.
ROW_GROUP_SIZE = 20_000

KEY_SPEC_VERSION = "chap.spec_version"
KEY_DATASET = "chap.dataset"
KEY_COLUMNS = "chap.columns"
KEY_PROVENANCE = "chap.provenance"
LOCATION_PROPERTIES = ("name", "parent", "level")


class DatasetFileError(ValueError):
    """A dataset file that cannot be read or does not follow the file format."""


class ColumnInfo(BaseModel):
    """What one observation column holds."""

    description: str | None = None
    unit: str | None = None
    source: str | None = None
    dhis2_data_element: str | None = Field(default=None, description="DHIS2 data element the values came from.")


class DatasetInfo(BaseModel):
    """Dataset settings carried by the observations file."""

    name: str | None = None
    target_column: str = "disease_cases"
    masked_ranges: list[MaskedRange] = Field(default_factory=list)


class DatasetFileMetadata(BaseModel):
    """The ``chap.*`` keys of a dataset file."""

    dataset: DatasetInfo = Field(default_factory=DatasetInfo)
    columns: dict[str, ColumnInfo] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)


def read_metadata(schema_metadata: dict[bytes, bytes] | None) -> DatasetFileMetadata:
    """Read the ``chap.*`` keys; a file without them gets the defaults. Other keys are left alone."""
    raw = {}
    for key, value in (schema_metadata or {}).items():
        if not key.startswith(b"chap."):
            continue
        name = key.decode(errors="replace")
        if len(value) > MAX_METADATA_BYTES:
            raise DatasetFileError(f"{name} is larger than {MAX_METADATA_BYTES} bytes")
        try:
            raw[name] = value.decode()
        except UnicodeDecodeError as e:
            raise DatasetFileError(f"{name} is not UTF-8 text") from e
    version = raw.get(KEY_SPEC_VERSION)
    if version is not None and version != SPEC_VERSION:
        raise DatasetFileError(f"{KEY_SPEC_VERSION} is {version!r}; this Chap reads version {SPEC_VERSION!r}")
    try:
        return DatasetFileMetadata(
            dataset=json.loads(raw.get(KEY_DATASET, "{}")),
            columns=json.loads(raw.get(KEY_COLUMNS, "{}")),
            provenance=json.loads(raw.get(KEY_PROVENANCE, "{}")),
        )
    except (json.JSONDecodeError, PydanticValidationError) as e:
        raise DatasetFileError(f"Invalid chap.* metadata: {e}") from e


def metadata_keys(metadata: DatasetFileMetadata) -> dict[bytes, bytes]:
    """The ``chap.*`` keys for ``metadata``, ready to merge into an Arrow schema's metadata."""
    keys = {
        KEY_SPEC_VERSION: SPEC_VERSION,
        KEY_DATASET: metadata.dataset.model_dump_json(),
        KEY_COLUMNS: json.dumps({name: info.model_dump(exclude_none=True) for name, info in metadata.columns.items()}),
        KEY_PROVENANCE: json.dumps(metadata.provenance),
    }
    return {key.encode(): value.encode() for key, value in keys.items()}


def _parquet_table(data: bytes, what: str) -> pa.Table:
    """Read a Parquet file after checking, in its footer, that it is not too large once decoded.

    The footer is written by the sender, so the check bounds honest and accidental files, and
    the row and size figures in the footer are what the decoder allocates for.
    """
    if not data.startswith(PARQUET_MAGIC):
        raise DatasetFileError(f"{what} is not a Parquet file")
    try:
        parquet = pq.ParquetFile(pa.BufferReader(data))
        footer = parquet.metadata
        if footer.num_rows > MAX_ROWS:
            raise DatasetFileError(f"{what} has {footer.num_rows} rows; at most {MAX_ROWS} are accepted")
        decoded = sum(footer.row_group(i).total_byte_size for i in range(footer.num_row_groups))
        if decoded > MAX_DECODED_BYTES:
            raise DatasetFileError(f"{what} decodes to {decoded} bytes; at most {MAX_DECODED_BYTES} are accepted")
        return parquet.read()
    except pa.ArrowException as e:
        raise DatasetFileError(f"{what} is not a readable Parquet file: {e}") from e


def read_observations(data: bytes) -> tuple[pd.DataFrame, DatasetFileMetadata]:
    """Read an observations file into a frame of ``location``, ``time_period`` and float feature columns."""
    table = _parquet_table(data, "observations")
    metadata = read_metadata(table.schema.metadata)
    names = table.schema.names
    missing = [name for name in ("location", "time_period") if name not in names]
    if missing:
        raise DatasetFileError(f"observations has no {' or '.join(missing)} column")
    not_numeric = [
        field.name
        for field in table.schema
        if field.name not in ("location", "time_period")
        and not (pa.types.is_integer(field.type) or pa.types.is_floating(field.type))
    ]
    if not_numeric:
        raise DatasetFileError(f"observations columns must be numeric; not numeric: {', '.join(not_numeric)}")
    if len(names) == 2:
        raise DatasetFileError("observations has no value columns")
    frame = table.to_pandas()
    for key in ("location", "time_period"):
        if frame[key].isna().any():
            raise DatasetFileError(f"observations has rows without a {key}")
        frame[key] = frame[key].astype(str)
    features = [name for name in names if name not in ("location", "time_period")]
    frame[features] = frame[features].astype(float)
    return frame, metadata


def read_locations(data: bytes) -> FeatureCollectionModel:
    """Read a locations file, GeoParquet or GeoJSON, into a feature collection keyed by ``location``."""
    if not data.startswith(PARQUET_MAGIC):
        try:
            return FeatureCollectionModel.model_validate_json(data)
        except PydanticValidationError as e:
            raise DatasetFileError(f"locations is neither GeoParquet nor valid GeoJSON: {e}") from e
    _parquet_table(data, "locations")  # a clear error for a broken file before geopandas reads it
    try:
        # geopandas reads file objects too; its stubs only list paths.
        frame = gpd.read_parquet(cast("Any", io.BytesIO(data)))
    except ValueError as e:
        raise DatasetFileError(f"locations is not GeoParquet: {e}") from e
    if "location" not in frame.columns:
        raise DatasetFileError("locations has no location column")
    if frame["location"].duplicated().any():
        raise DatasetFileError("locations has more than one row for the same location")
    if frame.crs is not None and frame.crs.to_epsg() != 4326:
        frame = frame.to_crs(4326)
    features = []
    for record in frame.to_dict("records"):
        geometry = record[frame.geometry.name]
        properties = {key: _json_value(record[key]) for key in LOCATION_PROPERTIES if key in frame.columns}
        features.append(
            {
                "type": "Feature",
                "id": str(record["location"]),
                "geometry": None if geometry is None else geometry.__geo_interface__,
                "properties": properties,
            }
        )
    try:
        return FeatureCollectionModel.model_validate({"type": "FeatureCollection", "features": features})
    except PydanticValidationError as e:
        raise DatasetFileError(f"locations has geometry Chap cannot use: {e}") from e


def _json_value(value: Any) -> Any:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return value.item() if hasattr(value, "item") else value


def write_dataset_files(
    dataset: DataSet, directory: Path, metadata: DatasetFileMetadata | None = None
) -> tuple[Path, Path]:
    """Write ``observations.parquet`` and ``locations.parquet`` for ``dataset`` into ``directory``."""
    metadata = metadata or DatasetFileMetadata()
    directory.mkdir(parents=True, exist_ok=True)
    features = list(dataset.field_names())
    frames = [
        pd.DataFrame(
            {
                "location": location,
                "time_period": [period.id for period in data.time_period],
                **{name: getattr(data, name) for name in features},
            }
        )
        for location, data in dataset.items()
    ]
    observations = pd.concat(frames, ignore_index=True).sort_values(["location", "time_period"], kind="stable")
    table = pa.Table.from_pandas(observations, preserve_index=False)
    table = table.replace_schema_metadata({**(table.schema.metadata or {}), **metadata_keys(metadata)})
    observations_path = directory / "observations.parquet"
    pq.write_table(table, observations_path, compression="zstd", row_group_size=ROW_GROUP_SIZE, write_page_index=True)

    locations_path = directory / "locations.parquet"
    rows = []
    for feature in dataset.polygons.features if dataset.polygons is not None else []:
        properties = feature.properties or {}
        rows.append(
            {
                "location": str(feature.id),
                **{key: properties.get(key) for key in LOCATION_PROPERTIES},
                "geometry": None if feature.geometry is None else shape(feature.geometry.model_dump()),
            }
        )
    frame = gpd.GeoDataFrame(
        rows, columns=["location", *LOCATION_PROPERTIES, "geometry"], geometry="geometry", crs=4326
    )
    # geopandas writes the `geo` key; Chap's keys are merged in afterwards so the file stays GeoParquet.
    frame.to_parquet(locations_path, index=False)
    table = pq.read_table(locations_path)
    table = table.replace_schema_metadata(
        {**(table.schema.metadata or {}), **metadata_keys(DatasetFileMetadata(provenance=metadata.provenance))}
    )
    pq.write_table(table, locations_path, compression="zstd")
    return observations_path, locations_path
