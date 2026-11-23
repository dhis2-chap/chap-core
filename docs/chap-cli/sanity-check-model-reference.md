# sanity-check-model Reference

`chap sanity-check-model` is a quick check that a model runs at all: it loads the model, trains it once and makes two forecasts, and fails if a forecast has missing values. It is much faster than `chap eval`, which trains and forecasts for every split of a backtest.

```console
chap sanity-check-model builtin:seasonal_median
chap sanity-check-model https://github.com/dhis2-chap/minimalist_example --dataset-csv ./data.csv
```

Inside a [Chap project](projects.md), it takes the model, dataset, column mapping, data settings, model configuration and run options from the project, so it runs without options:

```console
cd laos
chap sanity-check-model
```

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model-name` | The model, as for `chap eval`: a local directory, GitHub URL, chapkit service URL, `docker://<image>` or `builtin:<name>` | Required |
| `--dataset-csv` | Path or URL to the dataset CSV. A `.geojson` with the same name is loaded with it | A small example dataset |
| `--data-source-mapping` | JSON file mapping model covariate names to CSV column names | None |
| `--data-configuration` | JSON or YAML file with data settings, such as masked ranges | None |
| `--model-configuration-yaml` | YAML file with the model's options. Options the model does not have give a warning | None |
| `--run-config.*` | How the model runs, as for `chap eval`, for example `--run-config.is-chapkit-model` for a chapkit directory or service | |

A `docker://` image or a chapkit model in a local directory is started for the check and stopped afterwards.

The older names `--model-url`, `--dataset-path`, `--model-config-path` and `--use-local-environement` still work. `--use-local-environment` is the same as `--run-config.ignore-environment`.
