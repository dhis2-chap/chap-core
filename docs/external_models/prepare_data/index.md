# Prepare data for Chap

## 1) Chap data format

Chap models are designed to be interchangeable: any model that follows the Chap interface can be evaluated on any Chap-compatible dataset. For this to work, datasets must follow a common **CSV format** so that models know which columns to expect and how to parse time periods and locations.

The required columns in the CSV-file are:

| Column          | Description                                                      |
| --------------- | ---------------------------------------------------------------- |
| `time_period`   | Time period in `YYYY-MM` (monthly) or `YYYY-Wnn` (weekly) format |
| `location`      | Location identifier matching the GeoJSON features                |
| `disease_cases` | Observed case counts                                             |

Additional columns (e.g. `rainfall`, `mean_temperature`, `population`) are used as covariates by models that need them. A matching GeoJSON file with region polygons is optional but recommended for spatial visualizations.

## 2) Getting data

Choose the approach that fits your situation:

- [Use your own data](own-data.md) — convert an existing dataset to the Chap format
- [Use example data](example-data.md) — download a ready-made dataset to get started quickly
- [Use DHIS2 data](dhis2-data.md) — export data from a DHIS2 instance


## 3) Validating your data

Use the `chap validate` command to check that your CSV is Chap-compatible before running an evaluation.

### Basic validation

```console
chap validate --dataset-csv [YOUR-FOLDR-PATH].csv
# e.g chap validate --dataset-csv example_data/laos_subset.csv
```

This checks for:

- Required columns (`time_period`, `location`)
- Missing values (NaN) in covariate columns
- Location completeness (every location has the same set of time periods)

### Validate GeoJSON (if applicable)

If you have a GeoJSON file alongside your CSV, validate it with the `--dataset-geo` flag:

```console
chap validate --dataset-csv [YOUR-FOLDR-PATH].csv --dataset-geo [YOUR-FOLDR-PATH].geojson
#e.g. chap validate --dataset-csv example_data/laos_subset.csv --dataset-geo example_data/laos_subset.geojson
```

This additionally checks that all `location` values in the CSV have a matching feature in the GeoJSON.
