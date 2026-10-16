# Validating your data

Use the `chap validate` command to check that your CSV is Chap-compatible before running an evaluation.

### Basic validation

```console
chap validate --dataset-csv [YOUR-FOLDR-PATH].csv
```
*e.g chap validate --dataset-csv example_data/laos_subset.csv*

This checks for:

- Required columns (`time_period`, `location`)
- Missing values (NaN) in covariate columns
- Location completeness (every location has the same set of time periods)

### Validate GeoJSON (if applicable)

If a GeoJSON file with the same base name sits next to your CSV (for example `laos_subset.csv` and `laos_subset.geojson`), it is discovered and validated automatically by the same command:

```console
chap validate --dataset-csv [YOUR-FOLDR-PATH].csv
```
*e.g. chap validate --dataset-csv example_data/laos_subset.csv*

This additionally checks that all `location` values in the CSV have a matching feature in the GeoJSON.
