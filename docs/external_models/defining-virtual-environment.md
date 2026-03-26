# Defining the MLflow environment for your model

### MLflow environment

In the MLflow file you just created, you need to specify one environment CHAP will run your model within. The available options are uv (Python), renv environment (for R models) and docker. Below it is described how you set up each environment.

## Option 1) uv environment (Python)

Use `uv_env` to specify a pyproject.toml for uv-managed environments. This is useful for models that use [uv](https://docs.astral.sh/uv/) for dependency management:

```yaml
uv_env: pyproject.toml
```

Commands will be executed via `uv run`, which automatically handles the virtual environment. Make sure your model directory contains a valid `pyproject.toml` with dependencies specified. See the [example uv model](https://github.com/dhis2-chap/chap-core/tree/master/external_models/naive_python_model_uv) for a complete example.

??? example "Example MLproject file with uv"

    ```yaml
    name: my_model
    uv_env: pyproject.toml
    entry_points:
      train:
        parameters:
          train_data: str
          model: str
        command: "python main.py train {train_data} {model}"
      predict:
        parameters:
          model: str
          historic_data: str
          future_data: str
          out_file: str
        command: "python main.py predict {model} {historic_data} {future_data} {out_file}"
    ```

## Option 2) R enviroment

We reccomend to using renv to handle and specify dependenices for your R-models (link) When using renv, you only need to put this line in ML-project file

```yaml
renv_env: renv.lock
```

??? example "Example MLproject file with renv"

    ```yaml
    name: my_model
    renv_env: renv.lock
    entry_points:
      train:
        parameters:
          train_data: str
          model: str
        command: "Rscript train.R {train_data} {model}"
      predict:
        parameters:
          model: str
          historic_data: str
          future_data: str
          out_file: str
        command: "Rscript predict.R {model} {historic_data} {future_data} {out_file}"
    ```

## Option 3) Docker environment

When UV or renv is not sufficent to reliably reproduce the environment, its also possible to spesificy a dokcer-env to run the model within.

Use `docker_env` to specify a Docker image:

For example:
```yaml
docker_env:
  image: ghcr.io/dhis2-chap/docker_r_inla
```
