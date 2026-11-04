# Bring your existing model to Chap

This guide is for you if you already have a forecasting model in R or Python and want to run it in Chap. You keep your code. The guide has three stages:

1. **Run the model outside Chap.** Split it into a train and a predict script that read and write Chap's data format, and check that they work on example data.
2. **Make it Chap-compatible.** Add a short `MLproject` file and evaluate the model with `chap eval`.
3. **Package it as a Docker image**, so that it can run next to a Chap installation.

You do not need to install chapkit or run a Chap server, and you only need Docker for the last stage.

If you do not have a model yet, see [Build a new model from a template](mlproject_tutorial.md) instead.

!!! note "Train and predict scripts are a recommendation, not a requirement"
    The only strict requirement is that your model exposes the chapkit REST API, which Chap uses to send it data and get forecasts back. The train and predict scripts and the `MLproject` file in this guide are the easiest way to get there: chapkit reads them and provides the API for you. If you are comfortable writing a web service, you can implement the API yourself instead, in any language. See the [chapkit documentation](https://dhis2-chap.github.io/chapkit/) and its [API reference](https://dhis2-chap.github.io/chapkit/guides/ml-workflows/#api-reference).

## Before you start

Follow [Set up your machine](setup.md). You need `chap` installed, and R if your model is in R.

The [R template](https://github.com/chap-models/mlproject_template_r) and the [Python template](https://github.com/chap-models/mlproject_template_python) contain a small model that is already set up the way this guide describes. Keep the one for your language open: you will copy files from it.

## 1. Run the model outside Chap

### Split the model into a train and a predict script

Chap runs your model as two commands:

- **train** reads a CSV file with historic data and saves a trained model to a file.
- **predict** loads the trained model, reads the historic data and a CSV file with climate data for the future periods, and writes the forecasts to a CSV file.

Chap passes the file paths as command-line arguments. The templates show the smallest version of this: [train.r](https://github.com/chap-models/mlproject_template_r/blob/main/train.r) and [predict.r](https://github.com/chap-models/mlproject_template_r/blob/main/predict.r) in R, [train.py](https://github.com/chap-models/mlproject_template_python/blob/main/train.py) and [predict.py](https://github.com/chap-models/mlproject_template_python/blob/main/predict.py) in Python.

The input CSV files have the columns `time_period`, `location` and `disease_cases`, plus the covariates your model uses, such as `rainfall`, `mean_temperature` and `population`. See [Chap data format](prepare_data/index.md). The future data has the same columns, but no `disease_cases`.

The predict script writes a CSV file with `time_period`, `location` and the forecast in a `sample_0` column, one row per location and future period. A model that gives several samples writes `sample_1`, `sample_2` and so on.

### Run the scripts on example data

Copy the template's `input` folder, with [trainData.csv](https://github.com/chap-models/mlproject_template_r/blob/main/input/trainData.csv) and [futureClimateData.csv](https://github.com/chap-models/mlproject_template_r/blob/main/input/futureClimateData.csv), into your model's folder. Then run the scripts. For R:

```console
mkdir -p output
Rscript train.R input/trainData.csv output/model.bin
Rscript predict.R output/model.bin input/trainData.csv input/futureClimateData.csv output/predictions.csv
```

Open `output/predictions.csv` and check that it has a `sample_0` column.

If R says `there is no package called 'xgboost'` (or another package), install the package and run again:

```console
Rscript -e 'install.packages("xgboost")'
```

For a Python model, install the packages it needs in a virtual environment, and keep that environment active in the terminal for this stage and the next. See [Virtual environments](../kigali-workshop/kigali-webinar-series/session-2/virtual-environments.md) if this is new to you.

The templates have an `isolated_run` script that runs both steps. You can copy it and change the file names to make reruns quicker.

## 2. Make the model Chap-compatible

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
- `required_covariates` lists the dataset columns the model reads, besides `time_period`, `location` and `disease_cases`. Chap checks that a dataset has them before it runs the model.
- `command` is what Chap runs. Chap replaces the `{...}` placeholders with file paths. Use these names, since they are the ones Chap knows: `train_data`, `historic_data`, `future_data`, `model` and `out_file`. Put them in the order your scripts expect.

The file has no line that says how to install the model's packages. For now, the model runs with the R or Python and the packages you already have. You add that line at the end of this stage.

If your `MLproject` already has a `docker_env`, `conda_env` or `python_env` line, Chap cannot use it to run the model as a chapkit service. Remove the line, or keep it and add `--run-config.ignore-environment` to the `chap eval` command below.

### Evaluate the model with Chap

From your model's folder, run:

```console
chap eval --model-name . --dataset-csv https://raw.githubusercontent.com/dhis2-chap/chap-core/master/example_data/laos_subset.csv --output-file eval.nc --plot --run-config.as-chapkit --run-config.ignore-environment
```

- `--model-name .` means the model in the current folder.
- `--dataset-csv` is the data to evaluate on. The example is monthly data for a few provinces in Laos. You can also use your own data in the [Chap data format](prepare_data/index.md).
- `--run-config.as-chapkit` runs the model the same way a Chap server will run it.
- `--run-config.ignore-environment` runs the scripts with the R or Python and the packages you already have, as in stage 1.

Chap trains and tests the model on several periods of the data (a backtest) and writes the results to `eval.nc`. `--plot` writes a plot of the forecasts against the real case counts to `eval.html`. Open it in a browser.

### If something goes wrong

- **A package is missing:** install it as in stage 1 and run `chap eval` again.
- **A required covariate is missing:** the dataset does not have a column your `MLproject` lists under `required_covariates`. Use a dataset that has it, or remove it from the list if the model does not need it.
- **Running an MLproject with docker_env as a chapkit service is not supported:** see the end of [Write the MLproject file](#write-the-mlproject-file).
- **Something else:** add `--run-config.debug` to the `chap eval` command to see the output of your scripts.

### Record the model's packages

To run anywhere other than your machine, the model has to say which packages it needs. Add a line to the `MLproject` that Chap uses to install them:

- **R:** record the packages with [renv](https://rstudio.github.io/renv/) (`renv::init()`, then `renv::snapshot()`) and add `renv_env: renv.lock`.
- **Python:** list the packages in a `pyproject.toml` with uv (`uv init`, then `uv add <package>`) and add `uv_env: pyproject.toml`.

Run `chap eval` again without `--run-config.ignore-environment`. Chap now installs the packages itself before it runs the model. The Docker image in the next stage installs them the same way.

## 3. Package the model as a Docker image

To run next to a Chap installation, the model is packaged as a Docker image. You need [Docker](setup.md#4-docker-only-to-deploy) for this stage.

!!! tip "Let an AI coding agent do the work"
    AI coding agents, such as Claude Code, Codex or GitHub Copilot, are now good at making an existing model Chap-compatible, both stage 2 and this stage. Open the agent in your model's folder and point it to a model that already works with chapkit, and to the documentation. With an agent, you often do not need to do much by hand. For example:

    > Make this model Chap-compatible, following https://chap.dhis2.org/chap-modeling-platform/external_models/existing_model/. Use https://github.com/chap-models/mlproject_template_r as the example to copy from: add an `MLproject` file, record the packages with renv, and add the `Dockerfile`, `.dockerignore`, `compose.yml` and GitHub workflows. Then check that `chap eval --run-config.as-chapkit` works on the example data.

    Use [mlproject_template_python](https://github.com/chap-models/mlproject_template_python) for a Python model. The [chapkit documentation](https://dhis2-chap.github.io/chapkit/), and its [MLproject runner guide](https://dhis2-chap.github.io/chapkit/guides/mlproject-runner/) in particular, are also useful to give the agent. Check what the agent changed, and run `chap eval` yourself before you publish the model.

Copy these files from the template for your language into your model's folder:

| File | What it does | R | Python |
|---|---|---|---|
| `Dockerfile` | Builds the image from a chapkit base image that runs the `MLproject` as a chapkit service, and installs the packages from `renv.lock` or `pyproject.toml` | [Dockerfile](https://github.com/chap-models/mlproject_template_r/blob/main/Dockerfile) | [Dockerfile](https://github.com/chap-models/mlproject_template_python/blob/main/Dockerfile) |
| `.dockerignore` | Keeps local files such as your installed packages and `output/` out of the image | [.dockerignore](https://github.com/chap-models/mlproject_template_r/blob/main/.dockerignore) | [.dockerignore](https://github.com/chap-models/mlproject_template_python/blob/main/.dockerignore) |
| `compose.yml` | Builds and starts the image with one command | [compose.yml](https://github.com/chap-models/mlproject_template_r/blob/main/compose.yml) | [compose.yml](https://github.com/chap-models/mlproject_template_python/blob/main/compose.yml) |
| `.github/workflows/ci.yml` | Tests the image on GitHub | [ci.yml](https://github.com/chap-models/mlproject_template_r/blob/main/.github/workflows/ci.yml) | [ci.yml](https://github.com/chap-models/mlproject_template_python/blob/main/.github/workflows/ci.yml) |
| `.github/workflows/publish-docker.yml` | Publishes the image on GitHub | [publish-docker.yml](https://github.com/chap-models/mlproject_template_r/blob/main/.github/workflows/publish-docker.yml) | [publish-docker.yml](https://github.com/chap-models/mlproject_template_python/blob/main/.github/workflows/publish-docker.yml) |

To download them all at once, run this in your model's folder. For R:

```console
mkdir -p .github/workflows && for f in Dockerfile .dockerignore compose.yml .github/workflows/ci.yml .github/workflows/publish-docker.yml; do curl -fsSL -o $f https://raw.githubusercontent.com/chap-models/mlproject_template_r/main/$f; done
```

For Python, replace `mlproject_template_r` with `mlproject_template_python`.

The R `Dockerfile` expects the `renv.lock`, `.Rprofile` and `renv/` folder that `renv::init()` created in stage 2. The Python one expects the `pyproject.toml` that `uv init` created.

In `compose.yml`, you can rename the service from `mlproject_template_r` or `mlproject_template_python` to your model's name. Then build the image and start it:

```console
docker compose up --build
```

The model is now at http://localhost:9090. In another terminal, check that Chap can use it:

```console
chap eval --model-name http://localhost:9090 --run-config.is-chapkit-model --dataset-csv https://raw.githubusercontent.com/dhis2-chap/chap-core/master/example_data/laos_subset.csv --output-file eval.nc
```

### Publish the image

GitHub builds and publishes the image for you every time you push to the `main` branch. If your model is not on GitHub yet, create a new, empty repository on GitHub and push your folder to it, see [Git and GitHub](../kigali-workshop/kigali-webinar-series/session-2/git-github.md). Then commit the new files and push them:

```console
git add -A && git commit -m "Package the model as a Docker image" && git push
```

On GitHub, the **Actions** tab of your repository shows the workflows building, testing and publishing the image as `ghcr.io/<your-user>/<your-repo>:latest`. See [Publish the image](mlproject_tutorial.md#publish-the-image) in the template tutorial for what to check there.

## Next steps

- Run the model from the Chap modeling app, see [Running Your Own Model](../modeling-app/running-your-own-model.md).
- Make parts of the model configurable with `user_options`, see [Model Configuration Options](additional_configuration.md#model-configuration-options).
- Add author and contact details, see [Model Metadata](model_metadata.md).
