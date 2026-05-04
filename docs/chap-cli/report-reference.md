# report Command Reference

The `report` command runs a model's optional `report` entry point to produce a PDF document describing the trained model (diagnostics, fitted-effect plots, posterior summaries, or whatever the model author chooses to emit).

## Synopsis

```console
chap report <MODEL_PATH> <MODEL_ARTIFACT> <DATASET_CSV> <OUT_FILE> [OPTIONS]
```

## Description

`chap report`:

1. Loads the model template from a local MLProject directory or a GitHub URL.
2. Loads historic data from a CSV file (auto-discovering the matching `.geojson` if present).
3. Invokes the MLProject's `report` entry point with the supplied trained-model artifact and the historic data.
4. Writes the resulting PDF to `out_file`.

The model must declare a `report` entry point in its `MLproject` file. See [Additional Configuration → Report Entry Point](../external_models/additional_configuration.md#report-entry-point). Models that do not define `report` will fail with an explicit error.

## Required Parameters

| Parameter | Description |
|-----------|-------------|
| `model_path` | Path to an MLProject directory or a GitHub URL pointing to one |
| `model_artifact` | Path to a previously trained model artifact (the file `train` wrote) |
| `dataset_csv` | Path to a CSV file with historic data in the standard Chap format |
| `out_file` | Output path for the generated PDF report |

A GeoJSON file with the same base name as `dataset_csv` is auto-discovered if present.

## Run Configuration

The same `--run-config.*` flags as on `chap eval` are available — most relevant here:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--run-config.ignore-environment` | Skip automatic environment setup | false |
| `--run-config.debug` | Enable verbose debug logging | false |
| `--run-config.log-file` | Path to write log output | None |
| `--run-config.run-directory-type` | Directory handling: `latest`, `timestamp`, or `use_existing` | timestamp |

## Additional Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model-configuration-yaml` | Path to a YAML file with model-specific parameter values | None |

If the model declares `user_options` and the `report` entry point lists `model_config` as a parameter, the YAML file is forwarded to the report script the same way it is for `train` and `predict`.

## Example

```console
chap report \
    ./my_model \
    ./runs/my_model/latest/model \
    ./data/vietnam.csv \
    ./reports/vietnam_report.pdf
```

With debug logging:

```console
chap report \
    https://github.com/dhis2-chap/my_model \
    ./model_artifact \
    ./data.csv \
    ./report.pdf \
    --run-config.debug \
    --run-config.log-file ./report.log
```

## See Also

- [Additional Configuration → Report Entry Point](../external_models/additional_configuration.md#report-entry-point) — defining the entry point in `MLproject`
- [eval Reference](eval-reference.md) — sibling command that produces the trained-model artifact during backtesting
