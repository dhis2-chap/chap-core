# Validating your data

Use the `chap validate` command to check that your CSV is Chap-compatible before running an evaluation.

### Basic validation

```bash
chap validate --dataset-csv [YOUR-FOLDR-PATH].csv
```
[YOUR-FOLDR-PATH] could for example be: "example_data/laos_subset"

This checks for:

- Required columns (`time_period`, `location`)
- Missing values (NaN) in covariate columns
- Location completeness (every location has the same set of time periods)

### Validate GeoJSON (if applicable)

If you have a GeoJSON file alongside your CSV, validate it with the `--dataset-geo` flag:

```bash
chap validate --dataset-csv [YOUR-FOLDR-PATH].csv --dataset-geo [YOUR-FOLDR-PATH].geojson
```
[YOUR-FOLDR-PATH] could for example be: "example_data/laos_subset.csv"

This additionally checks that all `location` values in the CSV have a matching feature in the GeoJSON.
