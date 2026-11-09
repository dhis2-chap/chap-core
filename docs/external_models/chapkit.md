---
description: "How Chap runs models as chapkit REST services, the three ways to make a model, and the data format Chap sends to a chapkit model."
---

# Running models with chapkit

Chapkit is one of two ways chap runs models. Chap is gradually migrating models to chapkit; both chapkit and the older approach are supported.

In contrast to the older approach, where models are run directly through docker and pyenv wrappers inside chap, chapkit models are separate REST API services that chap interacts with through HTTP requests.

This has several advantages:

- There are no docker-in-docker problems where chap (which is inside docker) has to run docker commands to start model containers.
- We have a strict and clear API for how chap interacts with models, which makes it easier to develop and maintain.
- Models can easily be distributed through docker images or other means, as long as they implement the chapkit API. It is up to the model to define how it is deployed and run, but chapkit makes it easy to spin up a rest api and make a docker image for a model.

This document describes very briefly how to make a model compatible with chapkit, and how to use chapkit models in chap.


## How to make a model compatible with chapkit

The simplest way is to describe the model in an `MLproject` file and let chapkit serve it, with no service code. See [Bring your existing model to Chap](existing_model.md) or [Build a new model from a template](mlproject_tutorial.md). For writing a chapkit service directly, see the chapkit documentation: [https://dhis2-chap.github.io/chapkit/](https://dhis2-chap.github.io/chapkit/).

### Three ways to make a model

| | MLproject, built-in Chap runner | MLproject served by chapkit (recommended) | chapkit service |
|---|---|---|---|
| **You write** | Train and predict scripts, and an `MLproject` file | The same files | Train and predict code, a `main.py` that defines the service, and a Dockerfile |
| **How Chap runs it** | Chap reads the `MLproject` and runs the train and predict commands itself | chapkit reads the `MLproject` and serves the model as a REST service. Chap sends it data over HTTP | Your service runs on its own. Chap sends it data over HTTP |
| **Evaluate with** | `chap eval --model-name <directory or GitHub URL>` | `chap eval --model-name <directory or GitHub URL> --run-config.as-chapkit` | `chap eval --model-name <service URL> --run-config.is-chapkit-model` |
| **Add to a Chap installation** | Chap fetches the model from GitHub, see [Getting new models into the modeling app](../modeling-app/managing-model-templates.md) | Run a Docker image built from a chapkit base image. It registers itself with Chap | Run the service's Docker image. It registers itself with Chap |
| **Where to start** | [Connect your model to the Chap standard](describe_model.md) | [Bring your existing model to Chap](existing_model.md) or [Build a new model from a template](mlproject_tutorial.md) | The [chapkit documentation](https://dhis2-chap.github.io/chapkit/) |

The first two use the same `MLproject` format, so a model written for one usually runs with the other. The exception is the environment: chapkit sets up `uv_env` and `renv_env`, but not `docker_env`, `conda_env` or `python_env`. What differs is who runs the model: Chap itself, or chapkit as a separate service. The second and third run the same way in Chap. The difference is whether chapkit builds the service from your `MLproject`, or you write the service code. A model can move from the second to the third with `chapkit mlproject migrate`, see [MLproject Migrate](https://dhis2-chap.github.io/chapkit/guides/mlproject-migrate/) in the chapkit docs.


## Data format sent to chapkit models

Chap sends data to chapkit models via the REST API in a standardized format. The data is sent as JSON with a `columns` array and a `data` array (column-oriented format).

### Required columns

The following columns are always present in the training and prediction data:

| Column | Description |
|--------|-------------|
| `time_period` | Time period identifier (see format below) |
| `location` | Location identifier |
| `disease_cases` | Number of disease cases (training data only) |
| `rainfall` | Rainfall measurement |
| `mean_temperature` | Mean temperature |
| `population` | Population count |

### Time period format

The `time_period` column uses ISO-like string formats:

**Weekly data:**
```
2020-W01
2020-W02
2020-W52
2021-W01
```

The format is `YYYY-Wnn` where:
- `YYYY` is the ISO week year
- `W` indicates weekly data
- `nn` is the zero-padded week number (01-53)

**Monthly data:**
```
2020-01
2020-02
2020-12
2021-01
```

The format is `YYYY-MM` where:
- `YYYY` is the year
- `MM` is the zero-padded month number (01-12)

### Example training data (weekly)

```json
{
  "columns": ["time_period", "location", "disease_cases", "rainfall", "mean_temperature", "population"],
  "data": [
    ["2020-W01", "district_a", 150, 45.2, 28.5, 50000],
    ["2020-W02", "district_a", 142, 52.1, 27.8, 50000],
    ["2020-W01", "district_b", 89, 38.7, 29.1, 35000],
    ["2020-W02", "district_b", 95, 41.3, 28.9, 35000]
  ]
}
```

### Example training data (monthly)

```json
{
  "columns": ["time_period", "location", "disease_cases", "rainfall", "mean_temperature", "population"],
  "data": [
    ["2020-01", "district_a", 580, 180.5, 28.2, 50000],
    ["2020-02", "district_a", 620, 165.3, 27.9, 50000],
    ["2020-01", "district_b", 340, 155.8, 29.0, 35000],
    ["2020-02", "district_b", 365, 148.2, 28.7, 35000]
  ]
}
```

### Run info

Along with the data, Chap sends a `run_info` object containing runtime parameters:

```json
{
  "prediction_periods": 3,
  "additional_continuous_covariates": ["humidity"]
}
```

| Field | Description |
|-------|-------------|
| `prediction_periods` | Number of future periods to predict |
| `additional_continuous_covariates` | List of additional covariate columns in the data |

`run_info` is sent on both `$train` and `$predict`. At train time
`prediction_periods` is the horizon the model is going to be asked for, clamped to the
`min_prediction_periods` and `max_prediction_periods` the service declares; at predict
time it is the length of the future frame. Chap leaves the field out when it does not
know the horizon, and the service then falls back to the value in its stored
configuration.

Chapkit resolves the horizon from `run_info` first, then the future frame, then the
stored configuration, so your model reads it from `config.prediction_periods` as usual.
This requires chapkit 2.1.0 or newer; older versions ignore `run_info` at train time.
Chap sent the horizon as `prediction_length` before chapkit 2.1.0, and chapkit still
accepts that spelling.


## How to run a chapkit model from the command line

To test that the model works with Chap, use the `chap eval` command. Give the model's container image with `docker://` in front, and Chap starts the image as a container for the duration of the command and removes it afterwards. This needs Docker.

**Example:**

First save this model configuration to a file called `testconfig.yaml`:

```yaml
user_option_values:
  max_epochs: 2
```

Then evaluate the chtorch model from its image:

```console
chap eval --model-name docker://ghcr.io/dhis2-chap/chtorch:chapkit2-8f17ee3 --dataset-csv https://raw.githubusercontent.com/dhis2/climate-health-data/refs/heads/main/lao/chap_LAO_admin1_monthly.csv --output-file eval.nc --plot --backtest-params.n-splits 2 --backtest-params.n-periods 3 --model-configuration-yaml testconfig.yaml
```

If the model is already running as a service, for example one you started yourself with `docker run -p 5001:8000 <image>`, give its URL instead and add `--run-config.is-chapkit-model`:

```console
chap eval --model-name http://localhost:5001 --run-config.is-chapkit-model --dataset-csv ./data.csv --output-file eval.nc
```

A model from [Chap Marketplace](https://chap-marketplace.dhis2.org) can also be set up once in a [Chap project](../chap-cli/projects.md#chap-marketplace-models), which stores its verified image so that `chap eval` runs without options.


## How to use chapkit models in chap with the modeling app

Model services from the [Chap Marketplace](https://chap-marketplace.dhis2.org)
are installed with `chap-admin install-all`, which starts the services and
registers their templates and configurations with chap, so they appear in the
modeling app. See
[Install the Models](../modeling-app/fresh-installation.md#5-install-the-models)
for the commands.

To add a model service that is not in the marketplace, see
[Running Your Own Model](../modeling-app/running-your-own-model.md).

