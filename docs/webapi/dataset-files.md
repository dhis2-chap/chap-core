# Dataset files (Parquet)

A dataset can be uploaded as two files instead of JSON:

| File | Format | Holds |
| --- | --- | --- |
| `observations.parquet` | Parquet | One row per location and period: `location`, `time_period`, then one numeric column per feature (`disease_cases`, `population`, `rainfall`, ...) |
| `locations.parquet` | GeoParquet, or GeoJSON | One row per location: `location`, optional `name`, `parent` and `level`, and the geometry |

`location` is the org unit id in both files. `time_period` is a string such as `2024W12` (week)
or `202403` (month); the other spellings Chap reads, such as `2024-W12` and `2024-03`, are
accepted too. A missing value is a Parquet null. Every column other than `location` and
`time_period` must be numeric.

## Dataset settings in the file metadata

Parquet files carry key-value metadata. Chap reads these keys from the observations file:

| Key | Holds |
| --- | --- |
| `chap.spec_version` | `"1"`; a file with another version is refused |
| `chap.dataset` | JSON: `name`, `target_column` (default `disease_cases`), `masked_ranges` |
| `chap.columns` | JSON, per column: `description`, `unit`, `source`, `dhis2_data_element` |
| `chap.provenance` | JSON: where the data came from, its license, when it was retrieved |

All keys are optional. Without a name, the uploaded file's name is used. `dhis2_data_element`
becomes the dataset's data source for that column. `chap.provenance` is checked but not stored yet.

pandas drops Parquet metadata, so write the keys with pyarrow:

```python
import json
import tempfile
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

frame = pd.DataFrame(
    {
        "location": ["O6uvpzGd5pu", "O6uvpzGd5pu"],
        "time_period": ["202401", "202402"],
        "disease_cases": [12.0, None],
        "rainfall": [80.5, 120.0],
    }
)
table = pa.Table.from_pandas(frame, preserve_index=False)
chap_keys = {
    "chap.spec_version": "1",
    "chap.dataset": json.dumps({"name": "Monthly dengue", "target_column": "disease_cases"}),
    "chap.columns": json.dumps({"rainfall": {"unit": "mm", "source": "ERA5-Land"}}),
}
table = table.replace_schema_metadata(
    {**table.schema.metadata, **{key.encode(): value.encode() for key, value in chap_keys.items()}}
)
directory = Path(tempfile.mkdtemp())
pq.write_table(table, directory / "observations.parquet")
```

A Chap `DataSet`, for example one read from a CSV file with its GeoJSON next to it, can be
written as both files at once:

```python
from chap_core.file_io.dataset_files import DatasetFileMetadata, DatasetInfo, write_dataset_files
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

dataset = DataSet.from_csv("example_data/laos_subset.csv")
observations_path, locations_path = write_dataset_files(
    dataset, directory, DatasetFileMetadata(dataset=DatasetInfo(name="Laos monthly"))
)
```

## Uploading

`POST /v1/crud/datasets/files` takes the two files as a multipart form with the parts
`observations` and `locations`:

```console
curl -X POST "$CHAP_URL/v1/crud/datasets/files?dryRun=true" \
  -F observations=@observations.parquet \
  -F locations=@locations.parquet
```

It validates and imports the same way as `POST /v1/analytics/make-dataset`: the answer lists the
imported and rejected locations, `dryRun=true` only validates, and otherwise the answer holds the
id of the import job to follow at `/v1/jobs/{id}`.

Limits: each file up to 100 MB, an observations file up to 1 million rows and 512 MB once
decoded, and each `chap.*` metadata key up to 64 KB. Keys other than `chap.*` are ignored.

The upload also works through a DHIS2 route. DHIS2 2.42.3 and later forward each file twice
([DHIS2-22596](https://dhis2.atlassian.net/browse/DHIS2-22596)); Chap ignores the extra copies.
