# Running models with chapkit

Chapkit is a new experimental way of integrating and running models through chap.

In contrast to the current chap implementation, where models are run directly through docker and pyenv wrappers inside chap, chapkit models are separate REST API services that chap interacts with through HTTP requests.

This has several advantages:

- There are no docker-in-docker problems where chap (which is inside docker) has to run docker commands to start model containers.
- We have a strict and clear API for how chap interacts with models, which makes it easier to develop and maintain.
- Models can easily be distributed through docker images or other means, as long as they implement the chapkit API. It is up to the model to define how it is deployed and run, but chapkit makes it easy to spin up a rest api and make a docker image for a model.

This is still experimental and under development, but we have a working prototype with a few models already.

This document describes very briefly how to make a model compatible with chapkit, and how to use chapkit models in chap.


## How to make a model compatible with chapkit

This guide is not written yet, for now we refer to the chapkit documentation: [https://dhis2-chap.github.io/chapkit/](https://dhis2-chap.github.io/chapkit/).


## Data format sent to chapkit models

CHAP sends data to chapkit models via the REST API in a standardized format. The data is sent as JSON with a `columns` array and a `data` array (column-oriented format).

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

Along with the data, CHAP sends a `run_info` object containing runtime parameters:

```json
{
  "prediction_length": 3,
  "additional_continuous_covariates": ["humidity"]
}
```

| Field | Description |
|-------|-------------|
| `prediction_length` | Number of future periods to predict |
| `additional_continuous_covariates` | List of additional covariate columns in the data |


## How to run a chapkit model from the command line with chap evaluate

To test that the model is working with chap, you can use the `chap evaluate` command. Instead of a github url or model name, you simply specify the REST API url to the model and add --is-chapkit-model to the command to tell chap that the model is a chapkit model.

**Example:**

First save this model configuration to a file called `testconfig.yaml`:

```yaml
user_option_values:
  max_epochs: 2
```

Then start the chtorch command on port 5001:

```console
docker run -p 5001:8000 ghcr.io/dhis2-chap/chtorch:chapkit2-8f17ee3
```

Then we can run the following command to evaluate the model. Note the http://localhost:5001 url, which tells chap to look for the model at that url.

```console
chap evaluate --model-name http://localhost:5001 --dataset-name ISIMIP_dengue_harmonized --dataset-country vietnam --report-filename report.pdf --debug --n-splits=2 --model-configuration-yaml testconfig.yaml --prediction-length 3 --is-chapkit-model
```


## How to use chapkit models in chap with the modeling app

NOTE: This is experimental, and the way this is done might change in the future.

1) Add your model to a file compose-models.yml, pick a port for your model that is not used by other models
2) Add your model to a config file inside config/configured_models (e.g. config/configured_models/local_config.yaml), with the url pointing to your model, using the container name you specified in compose-models, e.g. http://chtorch:5001 (localhost will not work for communication between the chap worker container and your model container). Here is an example:
  ```yaml
- url: http://chtorch:8000
  uses_chapkit: true
  versions:
    v1: "/v1"
  configurations:
    debug:
      user_option_values:
          max_epochs: 2
```
3) Start both chap and your model by running `docker compose -f compose.yml -f compose-models.yml up --build --force-recreate`

Now, the model should show up in the modeling app.

