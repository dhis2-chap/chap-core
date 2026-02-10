# Make Python-models compatible with Chap

CHAP supports multiple environment options for Python models, including Docker, MLflow/Conda, and uv.

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

## Next steps

- For prediction length constraints and configurable model parameters, see [Additional Configuration](additional_configuration.md)