---
description: "Build a new Chap model from the R or Python MLproject template, evaluate it with chap eval, and publish it as a chapkit Docker image with GitHub Actions."
---

# Build a new model from a template

This tutorial takes you from a template to your own model running in Chap. You write a train and a predict script, in R or Python, and a short `MLproject` file that describes them. Chap runs the model as a [chapkit](chapkit.md) service, but you do not write any service code.

!!! info "What you do not need"
    - **No separate chapkit install.** Installing Chap also installs chapkit, and `chap eval` starts and stops the chapkit service for you. There is no `brew install chapkit`.
    - **No Chap server.** You do not need to clone chap-core or run its `docker compose up`. That stack runs Chap for the Modeling App, and is not needed to develop or evaluate a model.
    - **No `chapkit mlproject migrate`** and no `main.py`. They turn a model into a chapkit service project with its own service code, which most models do not need.

If you already have a model, follow [Bring your existing model to Chap](existing_model.md) instead.

The tutorial starts from one of two template repositories:

- [mlproject_template_python](https://github.com/chap-models/mlproject_template_python), using uv for dependencies
- [mlproject_template_r](https://github.com/chap-models/mlproject_template_r), using renv for dependencies

Both contain the same model: a linear regression from rainfall and temperature to disease cases in the same month. The model is not meant to be good. It only shows the steps.

The tutorial has three stages:

1. **Run the model outside Chap** on example data, while you develop it (steps 1 to 3).
2. **Evaluate it with Chap** (step 4).
3. **Package it as a Docker image** and publish it, so that it can run next to a Chap installation (step 5).

## Requirements

Follow [Set up your machine](setup.md). You need `chap` and, for the R template, R. Docker is only needed for the last step, deploying the model.

## 1. Create your repository from the template

Make your own copy of the template on GitHub, so that you can change it and push your changes:

1. Open the [Python template](https://github.com/chap-models/mlproject_template_python) or the [R template](https://github.com/chap-models/mlproject_template_r) on GitHub.
2. Click **Use this template**, then **Create a new repository**. Give it a name, for example `my_model`. GitHub creates `https://github.com/<your-user>/my_model`.
3. Clone your copy, with your GitHub user name and the name you chose:

    ```console
    git clone https://github.com/<your-user>/my_model
    cd my_model
    ```

Your copy has everything the tutorial needs, including the files that build and publish the Docker image in step 5. You do not need to write those yourself. If git and GitHub are new to you, see [Git and GitHub](../kigali-workshop/kigali-webinar-series/session-2/git-github.md).

## 2. The files

| File | Purpose |
|------|---------|
| `train.py` / `train.r` | Reads the training data and saves a trained model to a file |
| `predict.py` / `predict.r` | Loads the trained model and writes forecasts for the future periods |
| `MLproject` | Tells Chap how to run the scripts, which covariates the model uses, and how to set up its environment |
| `pyproject.toml` / `renv.lock` | The model's dependencies |
| `isolated_run.py` / `isolated_run.r` | Runs train and predict on the example data, without Chap |
| `Dockerfile` | Packages the model as a chapkit service |
| `compose.yml` | Builds and starts the Docker image with one command |
| `.github/workflows/` | Test the Docker image and publish it when you push to GitHub |
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

Develop and debug the model on the example data first.

**Python:**

```console
uv run python isolated_run.py
```

uv installs the packages listed in `pyproject.toml` the first time.

**R:**

```console
Rscript -e "renv::restore()"
Rscript isolated_run.r
```

The first command installs the packages listed in `renv.lock`.

Both scripts train the model on `input/trainData.csv`, saves it in `output/` and writes forecasts for `input/futureClimateData.csv` to `output/predictions.csv`.

Edit the scripts and rerun until the model does what you want. To add a dependency, add it to `pyproject.toml` (Python), or install it with `renv::install("pkg")` and record it with `renv::snapshot()` (R). If the model reads new dataset columns, add them to `required_covariates`.

## 4. Evaluate the model with Chap

From the template directory, run:

```console
chap eval --model-name . --dataset-csv input/trainData.csv --output-file output/eval.nc --plot --run-config.as-chapkit
```

Chap copies the model to a run directory, sets up its environment (`uv sync`, or `renv::restore()`), starts it as a chapkit service on a free local port, runs a backtest against it, and stops the service. The results are in `output/eval.nc`, and `--plot` writes an HTML plot next to it.

The example dataset is small and synthetic. For a more realistic run, use a dataset such as `example_data/laos_subset.csv` from the [chap-core repository](https://github.com/dhis2-chap/chap-core). `--model-name` also accepts a GitHub URL, so others can evaluate your model without cloning it. See the [eval command reference](../chap-cli/eval-reference.md#running-an-mlproject-as-a-chapkit-service) for all options.

To make parts of the model configurable, for example the number of lags, declare `user_options` in the `MLproject`. See [Model Configuration Options](additional_configuration.md#model-configuration-options). Other fields you can set, such as author and contact details, are listed in [Model Metadata](model_metadata.md).

## 5. Package the model as a Docker image

To run next to a Chap installation, the model is packaged as a Docker image. Your repository already has the files for this, so there is nothing to write: `Dockerfile` describes the image, `compose.yml` builds and starts it, and the workflows in `.github/workflows` build and publish it on GitHub. You need [Docker](setup.md#4-docker-only-to-deploy) for this step.

### Build and test the image on your machine

From your repository folder, run:

```console
docker compose up --build
```

The first build takes a few minutes. The model is then at http://localhost:9090. In another terminal, check that Chap can use it:

```console
chap eval --model-name http://localhost:9090 --run-config.is-chapkit-model --dataset-csv input/trainData.csv --output-file output/eval.nc
```

Stop it with `Ctrl+C`, or `docker compose down`.

??? info "What the Dockerfile does"
    It starts from a chapkit base image that runs an `MLproject` as a chapkit service, installs your dependencies, and copies your scripts in unchanged. The Python one installs the dependencies from `pyproject.toml`:

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
    COPY renv/ renv/
    RUN Rscript -e "renv::restore(prompt = FALSE)"
    COPY . .
    ```

    The dependencies are installed before the scripts are copied, so editing a script does not reinstall them. There are also base images with the tidyverse or INLA preinstalled, see [Running in a container](https://dhis2-chap.github.io/chapkit/guides/mlproject-runner/#running-in-a-container) in the chapkit docs.

### Publish the image

GitHub builds and publishes the image for you every time you push to the `main` branch. Commit your changes and push them:

```console
git add -A && git commit -m "Update the model" && git push
```

Then follow the build on GitHub:

1. Open your repository on GitHub and click the **Actions** tab. Two workflows run for your push:
    - **CI** builds the image, starts it and runs `chap eval` against it.
    - **Publish Docker image** publishes the image.
2. When both have a green check mark, the image is published as `ghcr.io/<your-user>/<your-repo>:latest`. It is listed under **Packages** on the right side of your repository's front page.
3. The first time, open the package and check its visibility. If it is private, change it to public under **Package settings**, so that a Chap installation can pull it.

To publish a fixed version as well, push a tag such as `v1.0.0`: `git tag v1.0.0 && git push --tags`. The image then also gets the tag `1.0.0`.

If a workflow fails, click it to see which step failed and its output.

To use the model from the Chap modeling app, run the image next to Chap. The service registers itself with Chap on startup. See [Running Your Own Model](../modeling-app/running-your-own-model.md).

## Next steps

- If you need service logic that an `MLproject` cannot express, `chapkit mlproject migrate` turns the repository into a full chapkit project with a generated `main.py`. See [MLproject Migrate](https://dhis2-chap.github.io/chapkit/guides/mlproject-migrate/) in the chapkit docs.
- [Chapkit](chapkit.md) describes the data Chap sends to a chapkit service.
