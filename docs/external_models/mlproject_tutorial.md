# Build a model from an MLproject file

This tutorial takes you from a train and a predict script on your own machine to a model that runs in Chap. You write the scripts and a short `MLproject` file that describes them. The model runs as a [chapkit](chapkit.md) service, but you do not write any service code: no `main.py`, Makefile or compose file.

The tutorial starts from one of two template repositories:

- [mlproject_template_python](https://github.com/chap-models/mlproject_template_python), using uv for dependencies
- [mlproject_template_r](https://github.com/chap-models/mlproject_template_r), using renv for dependencies

Both contain the same model: a linear regression from rainfall and temperature to disease cases in the same month. The model is not meant to be good. It only shows the steps.

## How this differs from other ways to make a model

There are three ways to make a model that Chap can run. This tutorial covers the second.

| | MLproject, built-in Chap runner | MLproject served by chapkit (this tutorial) | chapkit service |
|---|---|---|---|
| **You write** | Train and predict scripts, and an `MLproject` file | The same files | Train and predict code, a `main.py` that defines the service, and a Dockerfile |
| **How Chap runs it** | Chap reads the `MLproject` and runs the train and predict commands itself | chapkit reads the `MLproject` and serves the model as a REST service. Chap sends it data over HTTP | Your service runs on its own. Chap sends it data over HTTP |
| **Evaluate with** | `chap eval --model-name <directory or GitHub URL>` | `chap eval --model-name <directory or GitHub URL> --run-config.as-chapkit` | `chap eval --model-name <service URL> --run-config.is-chapkit-model` |
| **Add to a Chap installation** | Chap fetches the model from GitHub, see [Getting new models into the modeling app](../modeling-app/managing-model-templates.md) | Run a Docker image built from a chapkit base image. It registers itself with Chap | Run the service's Docker image. It registers itself with Chap |
| **Where to start** | [Connect your model to the Chap standard](describe_model.md) | This page | The [chapkit documentation](https://dhis2-chap.github.io/chapkit/) |

The first two use the same `MLproject` format, so a model written for one usually runs with the other. The exception is the environment: chapkit sets up `uv_env` and `renv_env`, but not `docker_env`, `conda_env` or `python_env`. What differs is who runs the model: Chap itself, or chapkit as a separate service. The second and third run the same way in Chap. The difference is whether chapkit builds the service from your `MLproject`, or you write the service code. A model can move from the second to the third with `chapkit mlproject migrate`, see [Next steps](#next-steps).

## Requirements

- **Python template:** [uv](https://docs.astral.sh/uv/)
- **R template:** R, with `Rscript` on your PATH. renv installs itself the first time you start R in the template directory.
- **Chap Core 2.5.0 or newer**, see [Install Chap Core](../chap-cli/chap-core-cli-setup.md). Until 2.5.0 is released, install it from GitHub:

    ```console
    uv tool install git+https://github.com/dhis2-chap/chap-core --python 3.13
    ```

- **Docker**, for the deployment step

## 1. Create your repository from the template

On the template's GitHub page, click **Use this template** to make your own copy, then clone it. To try the template without making a copy, clone it directly:

```console
git clone https://github.com/chap-models/mlproject_template_python
cd mlproject_template_python
```

For R, use `mlproject_template_r`.

## 2. The files

| File | Purpose |
|------|---------|
| `train.py` / `train.r` | Reads the training data and saves a trained model to a file |
| `predict.py` / `predict.r` | Loads the trained model and writes forecasts for the future periods |
| `MLproject` | Tells Chap how to run the scripts, which covariates the model uses, and how to set up its environment |
| `pyproject.toml` / `renv.lock` | The model's dependencies |
| `isolated_run.py` / `isolated_run.r` | Runs train and predict on the example data, without Chap |
| `Dockerfile` | Packages the model as a chapkit service |
| `input/` | A small example dataset |

The Python `MLproject` looks like this:

```yaml
name: mlproject_template_python

meta_data:
  display_name: MLproject Template (Python)
  description: Linear regression from rainfall and temperature to disease cases. A starting point for new models.

required_covariates:
  - rainfall
  - mean_temperature

uv_env: pyproject.toml

entry_points:
  train:
    parameters:
      train_data: str
      model: str
    command: "python train.py {train_data} {model}"
  predict:
    parameters:
      historic_data: str
      future_data: str
      model: str
      out_file: str
    command: "python predict.py {model} {historic_data} {future_data} {out_file}"
```

- `required_covariates` lists the dataset columns the model reads, besides `time_period`, `location` and `disease_cases`. Chap checks that a dataset has them before it runs the model.
- `uv_env` (or `renv_env: renv.lock` in the R template) says how to set up the model's environment.
- `entry_points` are the commands for training and predicting. Chap fills in the `{...}` placeholders with file paths. Use these parameter names, since they are the ones Chap knows: `train_data`, `historic_data`, `future_data`, `model`, `out_file`, and optionally `model_config` and `polygons`.

The data format is described in [Chap data format](prepare_data/index.md). The predict script writes one row per location and future period, with the forecast in a `sample_0` column. A model that gives several samples writes `sample_1`, `sample_2` and so on.

## 3. Run the model without Chap

Develop and debug the model on the example data first:

```console
uv run python isolated_run.py   # Python
Rscript isolated_run.r          # R
```

This trains the model on `input/trainData.csv`, saves it in `output/` and writes forecasts for `input/futureClimateData.csv` to `output/predictions.csv`.

Edit the scripts and rerun until the model does what you want. To add a dependency, add it to `pyproject.toml` (Python), or install it with `renv::install("pkg")` and record it with `renv::snapshot()` (R). If the model reads new dataset columns, add them to `required_covariates`.

## 4. Evaluate the model with Chap

From the template directory, run:

```console
chap eval --model-name . --dataset-csv input/trainData.csv --output-file output/eval.nc --plot --run-config.as-chapkit
```

Chap copies the model to a run directory, sets up its environment (`uv sync`, or `renv::restore()`), starts it as a chapkit service on a free local port, runs a backtest against it, and stops the service. The results are in `output/eval.nc`, and `--plot` writes an HTML plot next to it.

The example dataset is small and synthetic. For a more realistic run, use a dataset such as `example_data/laos_subset.csv` from the [chap-core repository](https://github.com/dhis2-chap/chap-core). `--model-name` also accepts a GitHub URL, so others can evaluate your model without cloning it. See the [eval command reference](../chap-cli/eval-reference.md#running-an-mlproject-as-a-chapkit-service) for all options.

To make parts of the model configurable, for example the number of lags, declare `user_options` in the `MLproject`. See [Model Configuration Options](additional_configuration.md#model-configuration-options). Other fields you can set, such as author and contact details, are listed in [Model Metadata](model_metadata.md).

## 5. Deploy the model

To run the model next to a Chap installation, package it as a Docker image. The template's `Dockerfile` starts from a chapkit base image that runs an `MLproject` as a chapkit service. Your train and predict scripts are copied in unchanged. The Python one installs the dependencies from `pyproject.toml`:

```dockerfile
FROM ghcr.io/dhis2-chap/chapkit-py-cli:latest
WORKDIR /work
COPY pyproject.toml ./
RUN uv pip install --python /app/.venv/bin/python -r pyproject.toml
COPY . .
```

The R one restores the packages in `renv.lock`:

```dockerfile
FROM ghcr.io/dhis2-chap/chapkit-r-cli:latest
WORKDIR /work
COPY .Rprofile renv.lock ./
COPY renv/activate.R renv/settings.json renv/
RUN Rscript -e "renv::restore(prompt = FALSE)"
COPY . .
```

The dependencies are installed before the scripts are copied, so editing a script does not reinstall them. There are also base images with the tidyverse or INLA preinstalled, see [Running in a container](https://dhis2-chap.github.io/chapkit/guides/mlproject-runner/#running-in-a-container) in the chapkit docs.

Build the image and start it:

```console
docker build -t my-model .
docker run --rm -p 9090:8000 my-model
```

Check that the running service works with Chap by evaluating it by URL:

```console
chap eval --model-name http://localhost:9090 --run-config.is-chapkit-model --dataset-csv input/trainData.csv --output-file output/eval.nc
```

To use the model from the Chap modeling app, publish the image to a registry and add it as a service next to Chap. The service registers itself with Chap on startup. See [Running Your Own Model](../modeling-app/running-your-own-model.md).

## Next steps

- If you need service logic that an `MLproject` cannot express, `chapkit mlproject migrate` turns the repository into a full chapkit project with a generated `main.py`. See [MLproject Migrate](https://dhis2-chap.github.io/chapkit/guides/mlproject-migrate/) in the chapkit docs.
- [Chapkit](chapkit.md) describes the data Chap sends to a chapkit service.
