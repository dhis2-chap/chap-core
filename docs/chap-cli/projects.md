# Chap Projects

A Chap project is a directory that holds a model choice, a dataset, the configuration files and the results of evaluating it. Inside a project, `chap eval` runs without any options.

## Create a project

`chap init` creates the project directory:

```bash
chap init --help
```

```console
chap init laos --model builtin:seasonal_median --dataset ./example_data/laos_subset.csv
cd laos
chap validate
chap eval
```

`chap validate` checks the dataset against the model, and `chap eval` runs the backtest. Each `chap eval` writes a new NetCDF file and an HTML plot to `output/`, named after the model and the time of the run, so you can compare runs afterwards:

```console
chap export-metrics --input-files output/*.nc --output-file metrics.csv
```

Chap commands find the project from the current directory or any directory below it, the same way `git` finds a repository.

`chap init` loads the model once to check that it can be found and to read its options. A local model directory is stored in `chap.yaml` relative to the project, so the two can be moved together.

## Chapkit models

`chap init` takes the same `--run-config` options as `chap eval` and stores the ones you give in `chap.yaml`, so every later command runs the model the same way:

```console
# A chapkit model in a local directory, started and stopped by each command
chap init study --model ../chapkit_minimalist_example_py --run-config.is-chapkit-model

# An MLproject model served as a chapkit service for the duration of each run
chap init study --model https://github.com/chap-models/mlproject_template_python --run-config.as-chapkit

# A chapkit service that is already running
chap init study --model http://localhost:8000
```

For a chapkit service, `input/model_config.yaml` is filled with the options the service declares.

## Project layout

| Path | Contents |
|------|----------|
| `chap.yaml` | The model and evaluation settings. Marks the project root. |
| `input/data.csv` | The dataset, copied in by `chap init --dataset`. |
| `input/data.geojson` | Region polygons, copied in when found next to the dataset. |
| `input/data_settings.yaml` | Optional data settings such as masked ranges, as for `--data-configuration`. |
| `input/column_mapping.json` | Optional covariate-to-column mapping, as for `--data-source-mapping`. |
| `input/model_config.yaml` | The model's options. `chap init --model` writes it with the model's default values when the model has options. |
| `output/` | Evaluation results. |
| `.chap/` | Chap's working copies of the model, in place of `runs/`. Listed in `.gitignore`. |

`chap init --data-settings`, `--column-mapping` and `--model-config` copy existing files to these names. You can also add or edit the files by hand: a file in `input/` is used whenever it exists.

## chap.yaml

Every key in `chap.yaml` is a command-line option, with the leading `--` dropped and dashes written as underscores. Nested options such as `--backtest-params.n-periods` become nested keys:

```yaml
model_name: builtin:seasonal_median

eval:
  backtest_params:
    n_periods: 3
    n_splits: 7
  plot: true
```

Top-level keys apply to every command that takes them. A section named after a command, such as `eval:`, applies to that command only. Relative paths are relative to the project root, so `dataset_csv: ../shared/vietnam.csv` uses a dataset outside the project instead of `input/data.csv`.

A misspelled option inside a known option or a command section, such as `backtest_params: {n_period: 3}`, stops the command with an error. A top-level key that no project command takes gives a warning.

## Which value is used

For each option, the first of these that gives a value wins:

1. The option on the command line.
2. The command's section in `chap.yaml`.
3. A top-level key in `chap.yaml`.
4. The matching file in `input/`.
5. Project defaults: a new file in `output/` for `--output-file`, and `.chap/runs` for `--run-config.runs-dir`.
6. The command's own default.

So `chap eval --backtest-params.n-periods 6` changes the forecast horizon for one run and leaves the project unchanged.

The files in `input/` belong to the project's own dataset and model. When `--dataset-csv` is given on the command line, `data_settings.yaml` and `column_mapping.json` are not used, and when `--model-name` is given, `model_config.yaml` is not used. This keeps masked ranges or model options written for the project from being applied to another dataset or model.

Each run logs the project it uses and where each value came from.

## Commands that use the project

`chap eval`, `chap validate` and `chap model schema` take their options from the project. Other commands work as they do outside a project.
