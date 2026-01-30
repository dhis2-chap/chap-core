# MLproject Configuration

**Note:** We have future plans of going away from using MLproject files for configuring models, and instead use the new chapkit framework. This document describes the current implementation using MLproject files.

MLproject files define model templates in CHAP. They specify the model name, execution environment, and entry points for training and prediction.

## MLproject File Structure

MLproject files use YAML format with the following fields:

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Model identifier |
| Environment | Yes (one of) | `docker_env`, `python_env`, `uv_env`, `renv_env`, or `rest_api_url` |
| `entry_points` | Yes | Train and predict commands |
| `user_options` | No | Configurable parameters exposed to users |
| `meta_data` | No | Display name, author, description, status |
| `required_covariates` | No | List of required covariate names |
| `min_prediction_length` | No | Minimum prediction horizon |
| `max_prediction_length` | No | Maximum prediction horizon |

Note that defining rest_api_url is experimental, and is used for using MLproject files to configure chapkit models that run via REST API calls.

### Example

From `external_models/naive_python_model_with_mlproject_file_and_docker/MLproject`:

```yaml
name: naive_python

docker_env:
  image: python:3.13

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
  some_option:
    title: some_option
    type: integer
    default: '10'
    description: "Some option for the model"
```

## Parsing Flow

The following describes how the chap-core codebase parses MLproject files from local paths or GitHub URLs, and how we internally represent the information.

### Local Files

`get_model_template_from_mlproject_file()` in `chap_core/models/utils.py`:

This function validates against `ModelTemplateConfigV2` using Pydantic and returns a `ModelTemplate` instance.

### GitHub URLs

`fetch_mlproject_content()` in `chap_core/external/github.py`:

1. Parses URL to extract owner, repo name, and commit/branch
2. Constructs raw GitHub URL: `https://raw.githubusercontent.com/{owner}/{repo}/{commit}/MLproject`
3. Fetches and returns the YAML content from the MLproject file.

## Class Representation

### Core Classes (`chap_core/external/model_configuration.py`)

- **`ModelTemplateConfigV2`** - Main config class that combines all MLproject fields. Inherits from `ModelTemplateConfigCommon` and `RunnerConfig`.

- **`RunnerConfig`** - Environment settings. This is used to define the environment in which the model will run. It includes one of the following fields:
  - `entry_points: EntryPointConfig`
  - `docker_env: DockerEnvConfig`
  - `python_env: str`
  - `uv_env: str`
  - `renv_env: str`

- **`EntryPointConfig`** - Contains `train` and `predict` commands as `CommandConfig` objects

- **`CommandConfig`** - Single command with `command: str` and optional `parameters: dict`

### Metadata Classes (`chap_core/database/model_templates_and_config_tables.py`)

- **`ModelTemplateMetaData`** - Display information: `display_name`, `author`, `description`, `author_assessed_status`, `organization`, `contact_email`, `citation_info`

- **`ModelTemplateInformation`** - Technical details: `supported_period_type`, `user_options`, `required_covariates`, `min_prediction_length`, `max_prediction_length`, `target`, `allow_free_additional_continuous_covariates`

## Database Storage

### ModelTemplateDB (`chap_core/database/model_templates_and_config_tables.py:47`)

Stores parsed MLproject data. Inherits from `ModelTemplateMetaData` and `ModelTemplateInformation`.

Key fields:
- `name: str` - Unique model identifier
- `source_url: str` - GitHub URL or local path
- `version: str` - Version string
- `archived: bool` - Whether the template is archived

### ConfiguredModelDB (`chap_core/database/model_templates_and_config_tables.py:65`)

Stores configured model instances with specific parameter values.

Key fields:
- `name: str` - Unique configuration name
- `model_template_id: int` - Foreign key to `ModelTemplateDB`
- `user_option_values: dict` - User-specified option values
- `additional_continuous_covariates: list` - Extra covariates for this configuration

## Runner Selection

`get_train_predict_runner_from_model_template_config()` in `chap_core/runners/helper_functions.py:17-96` selects the appropriate runner based on environment configuration:

| Environment Field | Runner Class |
|-------------------|--------------|
| `docker_env` | `DockerTrainPredictRunner` |
| `uv_env` | `UvTrainPredictRunner` |
| `renv_env` | `RenvTrainPredictRunner` |
| `python_env` | `MlFlowTrainPredictRunner` |
| None | `CommandLineTrainPredictRunner` |

The runner handles executing the train and predict commands in the appropriate environment.
