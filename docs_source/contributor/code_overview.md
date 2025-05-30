# Code overview

The following is a very brief overview of the main modules and parts of the chap-core code-base, which can be used as a starting point for getting to know the code:

## The chap command line interface

- The entry point can be found in `cli.py`. Note that there is also a file called `chap_cli.py` which is an old entry point that is not being used.
- The `cli.py` file defines commands like `chap evaluate` are defined.

By looking at the code in the `cli.py` file, you can see how the different commands are implemented, and follow the function calls to see what code is being used.

## The REST API

The REST API is the main entry point for the Modeling App, and supports functionality like training models, predicting, harmonizing data etc.

- The main entry point for the API is in `rest_api_src/v1/rest_api.py` (newer versions will have a different version number than v1).
- The API is built using the `fastapi` library, and we are currently using Celery to handle asynchronous tasks (like training a model).
- Celery is currently abstracted away using the `CeleryPool` and `CeleryJob` classes.

### More about the rest API endpoints

The main endpoints for the REST API are defined in `rest_api_src/v[some version number]/rest_api.py.

- crud: mainly just database operations
- analytics: A bad name (should be changed in the future). Bigger things that are written specifically to be used by the frontend. Things that are not used by the modelling app should be deleted in the future (e.g. prediction-entry)
- debug:
- jobs:
- default: used by the old Prediction app (will be taken away at some point)

We use pydantic models to define all input and return types in the REST API. See `rest_api_src/data_models.py`. We also use pydantic models to define database schemas (see `dataset_tables.py`). These models are overriden for the rest API if the REST API needs anything to be different. The database gives the objects IDs (if there is a primary key is default None). The overrides for the REST API have become a bit of mess and are defined many places. These should ideally be cleaned up and put in one file.

A messy thing about the database models is that many tables have an id field that has the same behavious. This could ideally be solved by a decorator look for fields that have that behavious and create three classes from it: One for the database, one for Read and one for Create, so that we don't need to do inheritance to get these classes. This has to be done by adding methods to get the classes.

### DB schemas:

Everything that inherits from `SqlModel` AND has `table=True` becomes a database table. The easiest way to find tables is to simply search for `table=True`

## External models

The codebase contains various abtractions for external models. The general idea is that an external model is defined by what commands
it uses to train and predict, and what kind of environment (e.g. docker) it needs to run these commands. CHAP then handles the necessary steps
to call these commands in the given environment with correct data files.

### Runners

The `TrainPredictRunner` class defines an interface that provides method for running commands for training and predicting for a given model.
The `DockerTrainPredictRunner` class is a concrete implementation that defines how to run train/predict-commands in a docker environment.

### External model wrapping

The `ExternalModel` class is used to represent an external model, and contains the necessary information for running the mode,
like the runner (an object of a subclass of `TrainPredictRunner`, the model name etc).

This class is rarely used directly. The easiest way to parse a model specification and get an object of `ExternalModel` is to
use the `get_model_from_directory_or_github_url` function. This function can take a directory or a github url, and parses the model specification
in order to get an `ExternalModel` object with a suitable runner. By following the code in this function, you can see how external models are loaded and run.

### Model evaluation and test/train splitting

A big nontrivial part of chap is to correctly split data into train and test sets for evaluation and passing these to models for evaluation.

A good starting point for understanding this process is the `evaluate_model` in the `prediction_evaluator.py` file. Functions like the `train_test_generator` function are relevant.
Currently, the main evaluation flow does not compute metrics, but simply plots the predictions and the actual values (in the `plot_forecasts` function).

## Models and ModelTemplates

The following is a draft mermaid notation overview:

```
flowchart TD


    ModelTemplate_get_config_class --> ModelConfiguration



     E[evaluate or predict or backtest]--> get_model_template_from_directory_or_github_url -->
    get_model_template_from_mlproject_file --> ModelTemplate

    ModelTemplate --> A[get_model with object of ModelConfiguratio] --> Model

    ModelTemplate --> B["get_default_model()"] --> Model
    ModelTemplate --> get_train_predict_runner --> TrainPredictRunner


    deprecated --> get_model_from_directory_or_github_url --> get_model_from_mlproject_file




    Runner --> CommandLineRunner

    TrainPredictRunner --> DockerTrainPredictRunner
    TrainPredictRunner --> CommandLineTrainPredictRunner --> CommandLineRunner
```
