Developer getting started
=========================

The main intended way of contributing to CHAP is by contributing with models, for which we have a modularized system that makes it easy to contribute.
For this, we have guides/tutorials (see the menu) that explain how to make models compatible with CHAP.

We are also working on adding similar guides for contributing with custom code for evaluating models and visualizing results.
The code for evaluating and visualizing results is currently tightly integrated into the chap-core code base, but the plan is to 
make this more modularized and easier to contribute to.

If you want to contribute to the chap-core code base itself, see the following guides:


Getting started working with the chap-core codebase
---------------------------------------------------

Contributing to the chap-core code base requires that you have a clone of the chap-core repository.

See the following guide for setting up the code base locally for development, 
and see the code overview guide for a brief overview of the code base.

Development setup
------------------

If you want to contribute to the chap-core Python codebase, you will need to set up a development environment. 
The following is our recommended setup. You will need to have Python 3.10 or a higher version installed.

1. Clone the chap-core repository:

.. code-block:: console

    $ git clone git@github.com:dhis2-chap/chap-core.git

2. Make sure you have `uv installed <https://docs.astral.sh/uv/getting-started/installation/>`_.

3. Install the dependencies. Inside the project folder, run:

.. code-block:: console

    $ uv sync --dev

Note that uv creates a virtual environment for you, so you don't need to create one yourself. This environment exists in the .venv directory. In order to run things through the virtual environment, you can use the `uv run` command. You can also activate the virtual environment manually with `source .venv/bin/activate`.

4. Run the tests to make sure everything is working:

.. code-block:: console

    $ uv run pytest

If the tests are passing, you are ready to start developing. Feel free to check out open issues in the chap-core Github repository. If you have any problems installing or setting up the environment, feel free to `contact us <https://github.com/dhis2-chap/chap-core/wiki>`_.


Contributing code
------------------

Code contributions should mainly happen by creating a pull request in the chap-core repository. In order to do this, you
will have to have a clone of the chap-core repository on github (which is possible for anyone with a github account).
If you want to push branches directly to the main chap-core repository, reach out to us about this.

Code overview
--------------

The following is a very brief overview of the main modules and parts of the chap-core code-base, which can be used as a starting point for getting to know the code:

- The chap command line interface:
    - The entry point can be found in `cli.py`.
    - This is where commands like `chap evaluate` are defined. 
    By looking at the code here, you can see how the different commands are implemented, and follow the function calls to see what code is being used.
- REST api:
    - This is the API that is used by the Prediction app, and supports a wide range of functionality (like training models, predicting, harmonizing data etc).
    - The main entry point for the API is in `rest_api_src/v1/rest_api.py` (newer versions will have a different version number than v1).
    - The API is built using the `fastapi` library, and we are currently using Celery to handle asynchronous tasks (like training a model).
    - Celery is currently abstracted away using the `CeleryPool` and `CeleryJob` classes. 
- External models:
    - The main function for parsing external models is the `get_model_from_directory_or_github_url`, which relies on abstractions for representing runners (like `TrainPredictRunner`).
    By following the code in this function, you can see how external models are loaded and run.
- Model evaluation and test/train splitting:
    - A big nontrivial part of chap is to correctly split data into train and test sets for evaluation and passing these to models for evaluation.
    - A good starting point for understanding this process is the `evaluate_model` in the `prediction_evaluator.py` file.
    Functions like the `train_test_generator` function are relevant. 
    - Currently, the main evaluation flow does not compute metrics, but simply plots the predictions and the actual values (in the `plot_forecasts` function).




