# Additional Configuration
Some additional model configuration must be specified, and some fields are optional. This will be explained below and it is relevant for both Python and R models.

## Specifying prediction length constraints

Include `min_prediction_length` and `max_prediction_length` in your model configuration to define how many time periods your model can predict ahead. When users need predictions beyond your `max_prediction_length`, CHAP automatically uses ExtendedPredictor to make iterative predictions (see [supporting functionality](supporting_functionality.md)).

## Model Configuration Options

You can define configurable parameters in your MLproject file using `user_options`. This allows users to customize model behavior when running your model, without modifying the model code itself.

### Schema structure

Each option in `user_options` has the following fields:

- `title`: Display name for the parameter
- `type`: One of `string`, `integer`, `number`, `boolean`, or `array`
- `description`: What the parameter does
- `default`: Optional default value. If omitted, the parameter is required

### Example MLproject with user_options

```yaml
name: my_model

docker_env:
  image: python:3.11

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

user_options:
  n_lag_periods:
    title: n_lag_periods
    type: integer
    default: 3
    description: "Number of lag periods to include in the model"
  learning_rate:
    title: learning_rate
    type: number
    description: "Learning rate for training (required)"
```

### Providing configuration values

Configuration values can be provided via the `--model-configuration-yaml` CLI flag when running `eval` or other commands:

```console
chap eval my_model data.csv results.nc --model-configuration-yaml config.yaml
```

The configuration YAML file should contain the parameter values:

```yaml
n_lag_periods: 5
learning_rate: 0.01
```

### Validation rules

- Options without a `default` value are required and must be provided
- Only options defined in `user_options` are allowed in the configuration file
- Values must match the specified type (e.g., integers for `integer` type)

### Examples in the codebase

See the following examples that use `user_options`:

- [naive_python_model_with_mlproject_file_and_docker](https://github.com/dhis2-chap/chap-core/tree/master/external_models/naive_python_model_with_mlproject_file_and_docker)
- [web_based_model](https://github.com/dhis2-chap/chap-core/tree/master/external_models/web_based_model)
