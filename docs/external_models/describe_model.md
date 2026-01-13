# Describing your model in our yaml-based format

To make your model chap-compatible, you need your train and predict endpoints (as discussed [here](train_and_predict.md)) need to be formally defined in a [YAML format](https://en.wikipedia.org/wiki/YAML) that follows the popular [MLflow standard](https://www.mlflow.org/docs/latest/projects.html#project-format).
Your codebase need to contain a file named `MLproject` that defines the following:
- An entry point in the MLproject file called `train` with parameters `train_data` and `model`
- An entry point in the MLproject file called `predict` with parameters `historic_data`, `future_data`, `model` and `out_file`

These should contain commands that can be run to train a model and predict the future using that model. The model parameter should be used to save a model in the train step that can be read and used in the predict step. CHAP will provide all the data (the other parameters) when running a model.

Here is an [example of a valid MLproject file](https://github.com/dhis2-chap/minimalist_example/blob/main/MLproject) (taken from our minimalist_example).

The MLproject file can specify a docker image, Python virtual environment, or uv-managed environment that will be used when running the commands.
An example of this is the [MLproject file](https://github.com/dhis2-chap/minimalist_example_r/blob/main/MLproject) contained within our minimalist_example_r.

## Environment options

### Docker environment
Use `docker_env` to specify a Docker image:
```yaml
docker_env:
  image: python:3.11
```

### MLflow/Conda environment
Use `python_env` to specify a conda/pip environment file (uses MLflow to manage):
```yaml
python_env: python_env.yml
```

### uv environment
Use `uv_env` to specify a pyproject.toml for uv-managed environments. This is useful for models that use [uv](https://docs.astral.sh/uv/) for dependency management:
```yaml
uv_env: pyproject.toml
```

Commands will be executed via `uv run`, which automatically handles the virtual environment. Make sure your model directory contains a valid `pyproject.toml` with dependencies specified. See the [example uv model](https://github.com/dhis2-chap/chap-core/tree/master/external_models/naive_python_model_uv) for a complete example.

Example MLproject file with uv:
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

