# Dataset files (Parquet)

A dataset can be uploaded as two files instead of JSON:

| File | Format | Holds |
| --- | --- | --- |
| `observations.parquet` | Parquet | One row per location and period: `location`, `time_period`, then one numeric column per feature (`disease_cases`, `population`, `rainfall`, ...) |
| `locations.parquet` | GeoParquet, or GeoJSON | One row per location: `location`, optional `name`, `parent` and `level`, and the geometry |

`location` is the location's id in both files (with DHIS2 data, the organisation unit id). `time_period` is a string such as `2024W12` (week)
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

The keys are file-level Parquet metadata, so write them with a library that keeps it, such as
pyarrow; pandas drops Parquet metadata when it reads or writes a file.

Example files of every size, including files Chap must reject, can be generated with
[chap-test-data](https://dhis2-chap.github.io/chap-test-data/).

## Uploading

`POST /v1/crud/datasets/files` takes the two files as a multipart form with the parts
`observations` and `locations`:

```console
curl -X POST "$CHAP_URL/v1/crud/datasets/files?dryRun=true" \
  -F observations=@observations.parquet \
  -F locations=@locations.parquet
```

Chap recognises each file by its content, not by the Content-Type of its part: the observations
must be Parquet (starting with the bytes `PAR1`), and the locations are read as GeoParquet when
they are Parquet and as GeoJSON otherwise. Clients rarely send a useful Content-Type for Parquet
(browsers and curl send `application/octet-stream`), and some DHIS2 versions remove it in routes,
so it is ignored. Sending `application/vnd.apache.parquet` and `application/geo+json` is still good
practice.

It validates and imports the same way as `POST /v1/analytics/make-dataset`, in two steps:

1. **Checking, in the request.** Chap reads both files and checks them before it answers. The
   answer lists the locations that will be imported and those rejected, with the reason. With
   `dryRun=true` the request stops here.
2. **Storing, in the background.** Otherwise the answer also holds the `id` of an import job, and
   a worker stores the dataset after the request has finished. Follow the job at `/v1/jobs/{id}`:
   when its status is `SUCCESS`, its result is the new dataset's id.

```console
curl -s -X POST "$CHAP_URL/v1/crud/datasets/files" \
  -F observations=@observations.parquet \
  -F locations=@locations.parquet
# {"id": "<job id>", "importedCount": 1500, "rejected": [], ...}
curl -s "$CHAP_URL/v1/jobs/<job id>"
# {"status": "SUCCESS", "result": 42, ...} once stored
```

Storing takes longer than checking for large datasets, about a minute at 1,500 locations of
weekly data over 7 years. The request itself does not wait for it.

Limits: each file up to 100 MB, an observations file up to 1 million rows, each file up to 512 MB
in memory once read (counting every repeated value in full), and each `chap.*` metadata key up to
64 KB. Other metadata keys are ignored, including the `pandas` key pandas writes, so a file written
from a DataFrame with an index reads the same as one without. Column names must be unique.

The upload also works through a DHIS2 route. DHIS2 2.42.3 and later forward each file twice
([DHIS2-22596](https://dhis2.atlassian.net/browse/DHIS2-22596)); Chap ignores the extra copies.
