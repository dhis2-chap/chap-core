
Code overview
--------------

The following is a very brief overview of the main modules and parts of the chap-core code-base, which can be used as a starting point for getting to know the code:

The chap command line interface
=================================

- The entry point can be found in `cli.py`. Note that there is also a file called `chap_cli.py` which is an old entry point that is not being used.
- The `cli.py` file defines commands like `chap evaluate` are defined. 

By looking at the code in the `cli.py` file, you can see how the different commands are implemented, and follow the function calls to see what code is being used.


The REST API
==============

The REST API is the main entry point for the Prediction app, and supports functionality like training models, predicting, harmonizing data etc. We are using FastAPI to make the API which has very good documentation here: https://fastapi.tiangolo.com/

- The main entry point for the API is in `rest_api_src/v1/rest_api.py` (newer versions will have a different version number than v1).
- The API is built using the `fastapi` library, and we are currently using Celery to handle asynchronous tasks (like training a model).
- Celery is currently abstracted away using the `CeleryPool` and `CeleryJob` classes. 

The Database
============
All database tables are defined in `database/tables.py`. We're using SQLModel to handle the database interactions which
has a very good documentation here: https://sqlmodel.tiangolo.com/

External models
================

The codebase contains various abstractions for external models. The general idea is that an external model is defined by what commands
it uses to train and predict, and what kind of environment (e.g. docker) it needs to run these commands. CHAP then handles the necessary steps
to call these commands in the given environment with correct data files.

Runners
_________

The `TrainPredictRunner` class defines an interface that provides method for running commands for training and predicting for a given model.
The `DockerTrainPredictRunner` class is a concrete implementation that defines how to run train/predict-commands in a docker environment.

External model wrapping
_________________________
 
The `ExternalModel` class is used to represent an external model, and contains the necessary information for running the mode, 
like the runner (an object of a subclass of `TrainPredictRunner`, the model name etc).

This class is rarely used directly. The easiest way to parse a model specification and get an object of `ExternalModel` is to 
use the `get_model_from_directory_or_github_url` function. This function can take a directory or a github url, and parses the model specification
in order to get an `ExternalModel` object with a suitable runner. By following the code in this function, you can see how external models are loaded and run.



Model evaluation and test/train splitting
___________________________________________

A big nontrivial part of chap is to correctly split data into train and test sets for evaluation and passing these to models for evaluation.

A good starting point for understanding this process is the `evaluate_model` in the `prediction_evaluator.py` file. Functions like the `train_test_generator` function are relevant. 
Currently, the main evaluation flow does not compute metrics, but simply plots the predictions and the actual values (in the `plot_forecasts` function).




