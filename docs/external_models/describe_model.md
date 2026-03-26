# Describing your model in our yaml-based format

To make your model chap-compatible, you need your train and predict endpoints (as discussed [here](train_and_predict.md)) need to be formally defined in a [YAML format](https://en.wikipedia.org/wiki/YAML) that follows the popular [MLflow standard](https://www.mlflow.org/docs/latest/projects.html#project-format).
Your codebase need to contain a file named `MLproject`. After adding the MLproject-file, you project may look like this:

```
my_model/
├── MLproject
├── input (input data, e.g. disease, climate)
├── output (predcitions / evaluations)
├── train.py (or R-file)
├── pyproject.toml (or R-file)
└── predict.py (or R-file)
```

## MLproject
- An entry point in the MLproject file called `train` with parameters `train_data` and `model`
- An entry point in the MLproject file called `predict` with parameters `historic_data`, `future_data`, `model` and `out_file`

These should contain commands that can be run to train a model and predict the future using that model. The model parameter should be used to save a model in the train step that can be read and used in the predict step. Chap will provide all the data (the other parameters) when running a model.

### Example MLproject file

Taken from our minimalist_example

```yaml
name: minimalist_example_uv

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

### Predict and train

Describe predict, train endpoint and how parameters got mapped to files paramters, how it works

### ML file environment

The MLproject file can specify a docker image, Python virtual environment, uv-managed environment, or renv environment (for R models) that will be used when running the commands.
An example of this is the [MLproject file](https://github.com/dhis2-chap/minimalist_example_r/blob/main/MLproject) contained within our minimalist_example_r.

## Next steps

- For Python-specific environment setup, see [Python-models](python_models.md)
- For R-specific environment setup, see [R-models](r_models.md)
