# simulate Command Reference

The `simulate` command runs a model's optional `simulate` entry point to generate disease data for the locations and periods in a covariate CSV.

## Synopsis

```console
chap simulate <MODEL_PATH> <DATASET_CSV> <OUT_FILE> [OPTIONS]
```

## Description

`chap simulate`:

1. Loads the model template from a local MLProject directory or a GitHub URL.
2. Loads covariates from a CSV file (auto-discovering the matching `.geojson` if present). The CSV does not need a `disease_cases` column.
3. Invokes the MLProject's `simulate` entry point.
4. Checks that the output covers the same locations and periods and contains the model's target column.
5. Writes the result to `out_file`, plus a `.geojson` next to it if the input had polygons.

The output is a regular Chap dataset and can be used with `chap eval`. See [Additional Configuration → Simulate Entry Point](../external_models/additional_configuration.md#simulate-entry-point) for how to define the entry point.

## Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model-configuration-yaml` | YAML file with `user_option_values`, e.g. simulation parameters or a seed | None |
| `--run-config.ignore-environment` | Skip automatic environment setup | false |
| `--run-config.debug` | Enable verbose debug logging | false |

## Example

```console
chap simulate \
    --model-name tests/fixture_models/climate_simulation_model \
    --dataset-csv ./covariates.csv \
    --out-file ./simulated.csv
```
