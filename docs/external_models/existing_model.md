---
description: "Make an existing R or Python forecasting model Chap-compatible: what Chap requires of a model, how to map messy or joint train-and-forecast code onto train and predict scripts, write the MLproject file and evaluate with chap eval --run-config.as-chapkit."
---

# Bring your existing model to Chap

This guide is for you if you already have a forecasting model in R or Python and want to run it in Chap. The model may be a single script, a notebook, or several files that download, clean, fit and forecast in one go. You keep your code and your modelling choices. What changes is how the code is called and how data goes in and out.

The guide has three parts:

1. **What Chap needs from your model.** The requirements, in one place.
2. **Reshape your code** into a train and a predict script that meet them, and check them on example data.
3. **Make it Chap-compatible.** Add a short `MLproject` file and evaluate the model with `chap eval`.

When `chap eval` works, [Package and publish your model](package_and_publish.md) takes it the rest of the way to a Chap installation. You need Docker only for that last step, and you never need a running Chap server to develop or evaluate a model.

If you do not have a model yet, see [Build a new model from a template](mlproject_tutorial.md) instead.

## Before you start

Follow [Set up your machine](setup.md). You need `chap` installed, and R if your model is in R.

The [R template](https://github.com/chap-models/mlproject_template_r) and the [Python template](https://github.com/chap-models/mlproject_template_python) contain the smallest possible model that meets the requirements below. They are not a starting point for your code, but a reference: when you are unsure what a finished script or `MLproject` looks like, compare with the one in the template for your language.

## 1. What Chap needs from your model

Chap never calls your model as a whole. It runs two commands, one after the other, and hands each of them file paths:

| Command | Files Chap passes to it | What it must write |
|---|---|---|
| `train` | `train_data`: a CSV file with historic data.<br>`model`: a path where the script may save a file. | Nothing is required. Most models fit here and save the fitted model to the `model` path. The script may also do nothing at all, see below. |
| `predict` | `model`: the path from `train`.<br>`historic_data`: the same historic data.<br>`future_data`: a CSV file with covariates for the periods to forecast.<br>`out_file`: the path to write the forecast to. | A CSV file at `out_file` with a forecast for every row in `future_data`. |

Chap writes the CSV files, fills in the paths and runs your commands. The order and form in which your scripts take the paths is up to you, see [Write the MLproject file](#write-the-mlproject-file).

**The data your model receives** is one CSV file in the [Chap data format](prepare_data/index.md): one row per location and time period, with the columns `time_period`, `location`, `disease_cases` and the covariates the model asks for, such as `rainfall`, `mean_temperature` and `population`. `time_period` is `2023-01` for monthly or `2023-W05` for weekly data. The future data has the same columns, without `disease_cases`. If your code expects other column names, a wide table, or dates as `Date` objects, the conversion becomes part of your scripts.

**The forecast your model writes** is a CSV file with `time_period`, `location` and a `sample_0` column, one row per row in the future data. A model that gives several samples from its predictive distribution writes `sample_1`, `sample_2` and so on, which Chap uses for uncertainty intervals and metrics. A model with a point forecast writes just `sample_0`.

**Rules that are easy to miss:**

- **One call covers all locations and all future periods.** The future data has every location in one file, usually three to twelve periods ahead. Your predict script forecasts all of them in one run. Code written for one district or for next month only needs a loop or a multi-step forecast.
- **The horizon is whatever is in the future data.** Do not hard-code the number of periods ahead. Count the rows per location.
- **No data of your own.** Everything the model uses comes from the two CSV files. The scripts cannot read files from your computer, download data, or query a database. If your model needs something Chap does not provide, such as a lookup table, put it in the model's folder and read it from there with a relative path.
- **`disease_cases` can be missing** in some rows of the historic data. Decide what the model does with them.
- **Training is repeated.** When Chap evaluates a model, it runs train and predict once per test period, each time with the data up to that period. Train should therefore run without manual steps and in reasonable time.
- **Nothing else matters.** Plots, printouts, reports and intermediate files are ignored. Only `out_file` is read.

!!! note "A train script that does nothing"
    For many R models, fitting and forecasting are one operation: the model is fitted on the historic rows and the future rows together, with the future cases marked as missing, and the fitted values for those rows are the forecast. INLA models work this way, and so do several models in the Chap marketplace. Such a model cannot save a fitted object in train and reuse it in predict.

    This is fine. The train script can be empty: it is called, does nothing and exits. The predict script receives the full historic data anyway, together with the future data, and does all the work. Programmers call a step that does nothing a no-op. The cost is that the model is fitted on every predict call, which Chap does once per test period in an evaluation. For most models this takes seconds to minutes and is not a problem.

!!! note "Scripts are a recommendation, not a requirement"
    Strictly, Chap requires only that the model answers a small web API, which Chap uses to send data and get forecasts back. The two scripts and the `MLproject` file are the easiest way to get there: a tool called chapkit, which comes with Chap, reads them and provides the API for you. If you are comfortable writing a web service, you can implement the API yourself, in any language. See the [chapkit documentation](https://dhis2-chap.github.io/chapkit/) and its [API reference](https://dhis2-chap.github.io/chapkit/guides/ml-workflows/#api-reference).

!!! tip "Let an AI coding agent do the work"
    AI coding agents, such as Claude Code, Codex or GitHub Copilot, are good at reshaping a model like this. Open the agent in your model's folder and give it this page and the template for your language. For example:

    > Make this model Chap-compatible, following https://chap.dhis2.org/chap-modeling-platform/external_models/existing_model/. Use https://github.com/chap-models/mlproject_template_r as the reference for what the result should look like: a train script, a predict script and an `MLproject` file. Keep the modelling as it is. Then check that `chap eval --run-config.as-chapkit` works on the example data.

    Use [mlproject_template_python](https://github.com/chap-models/mlproject_template_python) for a Python model. Read what the agent changed, and run `chap eval` yourself. Agents tend to silently change the model when the original code does not fit the two scripts, so check that the forecasts are the ones your model would give.

## 2. Reshape your code

### Map your code onto train and predict

Find the line in your code where the fitting ends and the forecasting starts. Where that line is decides which of these cases you are in:

- **Your code fits a model and then calls `predict` on it**, or does the equivalent. The train script runs everything up to the fit and saves the fitted object, with `saveRDS` in R or `pickle` or `joblib` in Python. The predict script loads it and runs the rest.
- **Fitting and forecasting are one step**, as in INLA and other models that are fitted with the future rows included. Make the train script do nothing and put all the code in the predict script, see the note above.
- **You fit one model per location**, in a loop. Keep the loop. In train, save a list or dictionary of fitted models keyed by location. In predict, loop over the locations in the future data and look each model up.
- **Your code is a notebook, R Markdown or Quarto file.** Move the code cells into a plain `.R` or `.py` file first. Drop the cells that make plots and tables.
- **Your code is a pipeline of several scripts**: download, clean, fit, forecast. The download step goes away, since Chap provides the data. Cleaning that depends on your data source also goes away. Cleaning that the model depends on, such as renaming columns, computing lags or reshaping, stays and is shared by both scripts, see below.

Whatever the case, the data preparation that both scripts need goes into one function in a shared file, `lib.R` or `lib.py`, that both scripts load. Then train and predict cannot drift apart.

Then go through the code and replace:

- every hard-coded file path with the path the script got as an argument;
- every column name with the one Chap uses, or add a rename at the top of the shared function;
- every reference to a fixed horizon with the number of rows per location in the future data;
- every plot, print of a table or report with nothing. Keep `print` or `message` calls that help you debug. Chap shows them with `--run-config.debug`.

### An example

A typical starting point is a script like this one. It reads two files from the author's computer, merges them, fits a Poisson regression and forecasts the months in a climate forecast file:

```r
library(dplyr)
cases <- read.csv("C:/Users/me/dengue/cases.csv")
climate <- read.csv("C:/Users/me/dengue/climate.csv")
df <- inner_join(cases, climate, by = c("district", "month"))
df$cases[is.na(df$cases)] <- 0
fit <- glm(cases ~ rain + temp, family = poisson, data = df)

future <- read.csv("C:/Users/me/dengue/climate_forecast.csv")
future$pred <- predict(fit, newdata = future, type = "response")
plot(future$month, future$pred, type = "l")
write.csv(future, "C:/Users/me/dengue/forecast.csv")
```

The code fits first and forecasts afterwards, so it is the first case above. The merge disappears, because Chap delivers one file. The column names change to Chap's, in one place. The plot goes. Both scripts take their file paths from the command line. The result is three files.

`lib.R` holds what both scripts need:

```r
prepare <- function(df) {
  # Chap's column names, mapped to the names the model code uses
  df$rain <- df$rainfall
  df$temp <- df$mean_temperature
  df
}
```

`train.R` fits the model and saves it:

```r
source("lib.R")
args <- commandArgs(trailingOnly = TRUE)
train_data <- args[1]
model_file <- args[2]

df <- prepare(read.csv(train_data))
df$disease_cases[is.na(df$disease_cases)] <- 0
fit <- glm(disease_cases ~ rain + temp, family = poisson, data = df)
saveRDS(fit, model_file)
```

`predict.R` loads it and forecasts every row in the future data:

```r
source("lib.R")
args <- commandArgs(trailingOnly = TRUE)
model_file <- args[1]
historic_data <- args[2]
future_data <- args[3]
out_file <- args[4]

fit <- readRDS(model_file)
future <- prepare(read.csv(future_data))
out <- future[, c("time_period", "location")]
out$sample_0 <- predict(fit, newdata = future, type = "response")
write.csv(out, out_file, row.names = FALSE)
```

This model does not use the historic data in predict, but it still receives it. A model with lagged cases would read it here to get the last observed values for each location.

A Python model follows the same shape, with `sys.argv` for the arguments, `pandas` for the CSV files and `joblib` for the model file. The [Python template](https://github.com/chap-models/mlproject_template_python) shows it.

### Run the scripts on example data

Copy the template's `input` folder, with [trainData.csv](https://github.com/chap-models/mlproject_template_r/blob/main/input/trainData.csv) and [futureClimateData.csv](https://github.com/chap-models/mlproject_template_r/blob/main/input/futureClimateData.csv), into your model's folder. Then run the scripts. For R:

```console
mkdir -p output
Rscript train.R input/trainData.csv output/model.bin
Rscript predict.R output/model.bin input/trainData.csv input/futureClimateData.csv output/predictions.csv
```

If R says `there is no package called 'xgboost'` (or another package), install the package and run again:

```console
Rscript -e 'install.packages("xgboost")'
```

For a Python model, install the packages it needs in a virtual environment, and keep that environment active in the terminal for this part and the next. See [Virtual environments](../kigali-workshop/kigali-webinar-series/session-2/virtual-environments.md) if this is new to you.

The templates have an `isolated_run` script that runs both steps. You can copy it and change the file names to make reruns quicker.

### Check the result

Before moving on, check `output/predictions.csv` against the requirements:

- It has the columns `time_period`, `location` and `sample_0`, and `sample_0` has no missing values.
- It has exactly one row for every row in `input/futureClimateData.csv`: every location, every period.
- Delete a few periods from the end of `input/futureClimateData.csv` and run predict again. The output shrinks with it. If it does not, the horizon is hard-coded somewhere.
- Blank out a few `disease_cases` values in `input/trainData.csv` and run both scripts again. They still run.
- Run train twice in a row. It does not wait for input and does not need anything that is not in the folder.

If a check fails, fix the scripts before you continue. `chap eval` runs these scripts many times and the error messages are further away from your code.

## 3. Make the model Chap-compatible

### Write the MLproject file

Create a file named `MLproject` (no file extension) in your model's folder. For an R model:

```yaml
name: my_model

required_covariates:
  - rainfall
  - mean_temperature

entry_points:
  train:
    parameters:
      train_data: str
      model: str
    command: "Rscript train.R {train_data} {model}"
  predict:
    parameters:
      historic_data: str
      future_data: str
      model: str
      out_file: str
    command: "Rscript predict.R {model} {historic_data} {future_data} {out_file}"
```

For a Python model, the commands are `python train.py ...` and `python predict.py ...`.

- `name` is a short name for your model, without spaces.
- `required_covariates` lists the dataset columns the model reads, besides `time_period`, `location` and `disease_cases`. Chap checks that a dataset has them before it runs the model. A model that uses population lists `population` here too.
- `command` is what Chap runs. Chap replaces the `{...}` placeholders with file paths. Use these names, since they are the ones Chap knows: `train_data`, `historic_data`, `future_data`, `model` and `out_file`. Put them in the order your scripts expect.

The file has no line that says how to install the model's packages. For now, the model runs with the R or Python and the packages you already have. You add that line at the end of this part.

If your `MLproject` already has a `docker_env`, `conda_env` or `python_env` line, Chap cannot use it to run the model as a chapkit service. Remove the line, or keep it and add `--run-config.ignore-environment` to the `chap eval` command below.

### Evaluate the model with Chap

From your model's folder, run:

```console
chap eval --model-name . --dataset-csv https://raw.githubusercontent.com/dhis2-chap/chap-core/master/example_data/laos_subset.csv --output-file eval.nc --plot --run-config.as-chapkit --run-config.ignore-environment
```

- `--model-name .` means the model in the current folder.
- `--dataset-csv` is the data to evaluate on. The example is monthly data for a few provinces in Laos. You can also use your own data in the [Chap data format](prepare_data/index.md).
- `--run-config.as-chapkit` runs the model the same way a Chap server will run it.
- `--run-config.ignore-environment` runs the scripts with the R or Python and the packages you already have, as in part 2.

Chap trains and tests the model on several periods of the data (a backtest) and writes the results to `eval.nc`. `--plot` writes a plot of the forecasts against the real case counts to `eval.html`. Open it in a browser.

### If something goes wrong

- **A package is missing:** install it as in part 2 and run `chap eval` again.
- **A required covariate is missing:** the dataset does not have a column your `MLproject` lists under `required_covariates`. Use a dataset that has it, or remove it from the list if the model does not need it.
- **Running an MLproject with docker_env as a chapkit service is not supported:** see the end of [Write the MLproject file](#write-the-mlproject-file).
- **The forecast is rejected or has gaps:** go back to [Check the result](#check-the-result). The usual causes are a missing location or period in the output, or `NA` values in `sample_0`.
- **Something else:** add `--run-config.debug` to the `chap eval` command to see the output of your scripts.

### Record the model's packages

To run anywhere other than your machine, the model has to say which packages it needs. Add a line to the `MLproject` that Chap uses to install them:

- **R:** record the packages with [renv](https://rstudio.github.io/renv/) (`renv::init()`, then `renv::snapshot()`) and add `renv_env: renv.lock`.
- **Python:** list the packages in a `pyproject.toml` with uv (`uv init`, then `uv add <package>`) and add `uv_env: pyproject.toml`.

Run `chap eval` again without `--run-config.ignore-environment`. Chap now installs the packages itself before it runs the model. The Docker image you build next installs them the same way.

??? note "If you have been reading the chapkit documentation"
    Chap runs your model through [chapkit](chapkit.md), which provides the REST API Chap talks to. You do not need to install chapkit separately: installing Chap installs it, and `chap eval` starts and stops the chapkit service for you. There is no `brew install chapkit`. You also do not need `chapkit mlproject migrate` or a `main.py`. They turn a model into a chapkit service project with its own service code, which most models do not need.

## Next steps

- [Package and publish your model](package_and_publish.md) as a Docker image, so that it can run next to a Chap installation and in the Chap Modeling App.
- Make parts of the model configurable with `user_options`, see [Model Configuration Options](additional_configuration.md#model-configuration-options).
- Add author and contact details, see [Model Metadata](model_metadata.md).
