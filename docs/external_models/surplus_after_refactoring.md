# Integrating External Models with DHIS2 through CHAP

Assuming you have CHAP running on a server with DHIS2 ([see this guide](../modeling-app/running-chap-on-server.md)), it is possible to
make new external models available.

Currently, CHAP has an internal registry of models that can be used.  
If you want to run a model that is not in the registry, this now has to be done by editing the local CHAP code at the server.  
However, we are working on making this more flexible. For now, please reach out if you want a new model to be added to the internal registry in a given installation.

The following figure shows how the train and predict entry points are part of a data flow between DHIS2 and the external model:

![External model integration with DHIS2](dhis_chap_integration.jpg)


# Overview of supported models
## Autoregressive weekly
This model has been developed internally by the Chap team. Autoregressive weekly is a deep learning model based on DeepAR that uses rainfall and temperature as climate predictors and models disease counts with a negative binomial distribution where the parameters are estimated by a recurrent neural network. The current model available through Chap can be [found here](https://github.com/knutdrand/weekly_ar_model).

## Epidemiar
Epidemiar is a Generalizes additive model (GAM) used for climate health forecasts. It requires weekly epidemilogical data, like disease cases and population, and daily enviromental data. As most of the data in CHAP is monthly or weekly we pass weakly data to the model, and then naively expand weekly data to daily data, which the epidemiar library again aggregates back to weekly data. The model produces a sample for each location per time point with and upper and lower boundary for some unknown quantiles. For more information regarding the model look [here](https://github.com/dhis2-chap/epidemiar_example_model). An example of running the model from the CHAP command line interface is
```
chap evaluate --model-name https://github.com/dhis2-chap/epidemiar_example_model --dataset-csv LOCAL_FILE_PATH/laos_test_data.csv --report-filename report.pdf --debug --n-splits=3
```
which requires that `laos_test_data` is saved locally, but we are working on making weekly datasets available internaly in CHAP.

## EWARS
EWARS is a Bayesian hierarchical model implemented with the INLA library. We use a negative binomial likelihood in the observation layer and combine several latent effect, both spatial and temporal, in the latent layer. The latent layer is log-transformed and scaled by the population, so it effectivaly models the proprotion of cases in each region. Specifically the latent layers combine a first order cyclic random walk to capture the seasonal effect, this is also included in the lagged exogenous variables rainfall and temperature, then a spatial smoothing with an ICAR and an iid effect to capture the spatial heterogeneity. The ICAR and iid can also be combined, scaled and reparameterized to the BYM2 model. Further information is available in the [model repository](https://github.com/dhis2-chap/chap_auto_ewars). An example of running the model from the CHAP command line interface is
```
chap evaluate --model-name https://github.com/dhis2-chap/chap_auto_ewars --dataset-name ISIMIP_dengue_harmonized --dataset-country vietnam --report-filename report.pdf --debug --n-splits=3
```

## ARIMA
A general ARIMA model is a timeseries model with an autoregressive part, a moving average part and the option to difference the original timeseries, often to make it stationary. Additonally we have lagged rainfall and temperature, which actually makes this an ARIMAX model, where the X indicates exogenous variables. This model handles each region individually and it expects monthly data for all the covariates. The model utilizes the `arima` function which chooses the order of the differencing, autoregression and the moving average for us. Further information is available in the [model repository](https://github.com/dhis2-chap/Madagascar_ARIMA). An example of running the model from the CHAP command line interface is
```
chap evaluate --model-name https://github.com/dhis2-chap/Madagascar_ARIMA --dataset-name ISIMIP_dengue_harmonized --dataset-country vietnam --report-filename report.pdf --debug --n-splits=3
```


# Wrapping GluonTS models

GluonTS provides a set of models that can be used for probabilistic time-series forecasting.
Here, we show how we can wrap these models into CHAP models, to enable using them on spatio-temporal data and to evalutate them against other models.

We will use the `DeepAREstimator` model from GluonTS, which is a deep learning model based on an RNN architecture. For this simple example we use
a model that does not take weather into account, but only the the auto-regressive time series data.
Let's start by loading the data and the model.

```python
from chap_core.file_io.example_data_set import datasets
from chap_core.adaptors.gluonts import GluonTSEstimator
from gluonts.torch import DeepAREstimator
from gluonts.torch.distributions import NegativeBinomialOutput

# Load the data
data = datasets['ISIMIP_dengue_harmonized'].load()['vietnam']

# Define the DeepAR model
n_locations = len(data.locations)
prediction_length = 4
deep_ar = DeepAREstimator(
    num_layers=2,
    hidden_size=24,
    dropout_rate=0.3,
    num_feat_static_cat=1,
    scaling=False,
    embedding_dimension=[2],
    cardinality=[n_locations],
    prediction_length=prediction_length,
    distr_output=NegativeBinomialOutput(),
    freq='M')

# Wrap the model in a CHAP model
model = GluonTSEstimator(deep_ar, data)
```

The model now is a chap compatible model and we can run our evaluation pipeline on it.

```python
from chap_core.assessment.prediction_evaluator import evaluate_model

evaluate_model(model, data, prediction_length=4, n_test_sets=8, report_filename='gluonts_deepar_results.csv')
```


## Running an external model in Python

CHAP contains an API for loading models through Python. The following shows an example of loading and evaluating three different models by specifying paths/github urls, and evaluating those models:

```python
import pandas as pd

from chap_core.assessment.prediction_evaluator import evaluate_model
from chap_core.models.utils import get_model_from_directory_or_github_url
from chap_core.file_io.file_paths import get_models_path
from chap_core.file_io.example_data_set import datasets
import logging

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    models_path = get_models_path()
    model_names = {
        #'deepar': models_path / 'deepar',
        'naive_model': models_path / 'naive_python_model_with_mlproject_file',
        # 'ewars': 'https://github.com/sandvelab/chap_auto_ewars'
    }

    dataset = datasets['ISIMIP_dengue_harmonized'].load()
    dataset = dataset['vietnam']
    n_tests = 7
    prediction_length = 6
    all_results = {}
    for name, model_name in model_names.items():
        model = get_model_from_directory_or_github_url(model_name)
        results = evaluate_model(model, dataset,
                                 prediction_length=prediction_length,
                                 n_test_sets=n_tests,
                                 report_filename=f'{name}_{n_tests}_{prediction_length}_report.pdf')
        all_results[name] = results

    report_file = 'evaluation_report.csv'
    df = pd.DataFrame([res[0] | {'model': name} for name, res in all_results.items()])
    df.to_csv(report_file, mode='w', header=True)

```

# Docker Compose (CHAP Core)

Starting CHAP Core using Docker Compose is specifically for those who want to use the CHAP Core REST-API, either together with other services or with the Modeling App installed on a DHIS2 server. See documentation for [Modeling App](modeling-app/modeling-app.md) for instructions on how to install the Modeling App.

**Requirements**

- Access to credentials for Google Earth Engine. (Google Service Account Email and Private Key)

## 1. Install Docker (if not installed)

**Docker** is a platform for developing, shipping, and running applications inside containers.

To download and install Docker, visit the official Docker website: [https://docs.docker.com/get-started/get-docker](https://docs.docker.com/get-started/get-docker)

## 2. Clone CHAP Core GitHub-Repository

You need to clone the CHAP Core repository from GitHub. Open your terminal and run the following command:

```sh
git clone https://github.com/dhis2-chap/chap-core.git
```

## 3. Add Credentials for Google Earth Engine

1. Open your terminal and navigate to the "chap-core" repository you cloned:

   ```sh
   cd chap-core
   ```

2. Open the "chap-core" repository in your code editor. For example, if you are using Visual Studio Code, you can use the following command in the terminal:

   ```sh
   code .
   ```

3. In your code editor, create a new file at the root level of the repository and name it `.env`.

4. Add the following environment variables to the `.env` file. Replace the placeholder values with your actual Google Service Account credentials:

   ```bash
   GOOGLE_SERVICE_ACCOUNT_EMAIL="your-google-service-account@company.iam.gserviceaccount.com"
   GOOGLE_SERVICE_ACCOUNT_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----<your-private-key>-----END PRIVATE KEY-----"
   ```

## 4. Start CHAP Core

At the root level of the repository (the same level you placed the .env-file), run:

```sh
docker-compose up
```

**This will build three images and start the following containers:**

- A REST-API (FastAPI)
- A Redis server
- A worker service

You can go to [http://localhost:8000/docs](http://localhost:8000/docs) to verify that the REST-API is working. A Swagger page, as shown below, should display:

![Swagger UI](../_static/swagger-fastapi.png)

## 5. Stop CHAP Core

```sh
docker-compose down
```

## Logs

When running things with docker compose, some logging will be done by each container. These are written to the `logs`-directory, and can be useful for debugging purposes:

- `logs/rest_api.log`: This contains logs part of the chap-core rest api
- `logs/worker.log`: This contains logs from the worker running the models. This should be checked if a model for some reason fails
- `logs/tas_{task_id}.log`: One log file is generated for each task (typically model run). The task_id is the internal task id for the Celery task.

# Data requirements in CHAP

CHAP expects the data to contain certain features as column names of the supplied csv files. Specifically, time_period, population, disease_cases, location, rainfall and mean_temperature. CHAP gives an error if any of these are missing in the supplied datafile. Additionally, there are conventions for how to represent time in the time_period. For instance, weekly data should be represented as

|time_period|
|-----------|
|2014-W52|
|2015-W01|
|2015-W02|
|2015-W03|

And for monthly data it should be

|time_period|
|-----------|
|2014-12|
|2015-01|
|2015-02|
|2015-03|

This requirement is checked for supplied data files for training and predicitng and also for the output data from the model. Additionaly the model should give samples from a distribuition, preferable $1000$ samples for each location and time index with column names `sample_0, sample_1` and so on.

A useful tool for handling data is the adapters that can be included in the MLproject file. These adapters map the internal names in CHAP to whatever you want them to be. For instance disease_cases could be mapped to cases, as in the MLproject file in the repository under the [dhis2-chap organization](https://github.com/dhis2-chap/chap_auto_ewars_weekly). In practice, the adapters copy the mentioned column and gives it the new column name.