.. _developing_custom_models:

Developing custom models with CHAP
==================================

CHAP is designed to allow model developers to easily develop their own models outside CHAP and use CHAP to benchmark/evaluate their models, or to import and use utility functions from CHAP in their own models.

We here provide guides for implementing custom models in *Python* and *R*. The recommended flow is slightly different for the two different languages, but the general idea is the same.


Developing custom models in Python
----------------------------------

Code base structure
....................

We recommend that you develop your model through a custom Python project and not inside the CHAP codebase. Your Python code should have command line entry points for *training* the model and *predicting* based on a trained model. This could e.g. simply be two Python files that are run with some command line arguments or a command line interface (e.g. built with something like argparse or typer).

Your code base should as a minimum have:

- An entry point for training the model (e.g. a file called train.py)
- An entry point for predicting based on a trained model (e.g. a file called predict.py)
- An MLProject configuration file for your model that specifies the entry points (se the section about integration with CHAP below)

An easy way to get started is to clone our barebone template repository for a Python model, which can be found `here <https://github.com/dhis2-chap/chap_model_template>`_. This will give you a train.py and predict.py file that you can use as starting points, as well with an MLProject configuration file. For a complete, runnable example, please see to the "minimalist_example" repository.

Step 1: Test/develop your model outside CHAP
.............................................

The following is a suggested workflow that can be used when developing and testing your model. For ease of development, we recommend a workflow where you can run your model without fully integrating it with CHAP first. This makes it easier to debug and test your model in isolation. You should still make sure your model handles the data formats that CHAP uses. The easiest way is to test directly on example data provided by CHAP. You can find such `example data here <https://github.com/dhis2-chap/chap-core/tree/dev/Minimalist_multiregion_example_data>`_. When using the template repository, the example data is already included in the file "trainData.csv"

Look at our `minimalist example <https://github.com/dhis2-chap/minimalist_example>`_ and `minimalist example with multiple regions <https://github.com/dhis2-chap/minimalist_multiregion>`_ to see an examples of "train.py" that trains a model on chosen climate data and a "predict.py" that forecasts future disease cases.

Run the isolated_run.py that comes with the template, and make sure you are able to train your model and generate samples for predictions before moving on to the next step. Make sure you have output columns called `sample_0`, `sample_1`, etc. in your predictions file.


Step 2: Running your model through CHAP
.........................................

If your model is able to generate samples to a csv file as shown above, it should be fairly easy to run the model through the CHAP command line interface. Make sure you have `chap-core <https://dhis2-chap.github.io/chap-core/installation.html>`_  installed before continuing.

The benefit of running your model through CHAP is that you can let CHAP handle all the data processing and evaluation, and you can easily compare your model to other models. To do this, you need to create an MLProject configuration file for your model. This file should specify the entry points for training and predicting, as well as any dependencies your model has. You can then run your model through CHAP using the CHAP CLI. If you use the `template reposity <https://github.com/dhis2-chap/chap_model_template>`_ this file will already be included, and should function immediately upon installation.

You can also see an example of such a file in the `minimalist example <https://github.com/dhis2-chap/minimalist_example>`_

The important part here is that the entry points for train and predict give commands that work with your model. These will be run inside the directory containing your model code. If creating a train.py and predict.py like shown above, you need to make sure these can be run through the command line.

Place this MLProject file in the same directory as your model code. You can then run your model through CHAP using the following command:

.. code-block:: console

    $ chap evaluate --model-name /path/to/your/model/directory --dataset-name ISIMIP_dengue_harmonized --dataset-country brazil --report-filename report.pdf --ignore-environment  --debug

Note the `--ignore-environment`. This means that we don't ask CHAP to use Docker or a Python environment when running the model. Instead the model will be run directly using the current environment you are in. This usually works fine when developing a model, but requires you to have both chap-core and the dependencies of your model available. The next step shows how to run your model in an isolated environment.

If the above command runs without any error messages, you have successfully evaluated your model through CHAP, and a file `report.pdf` should have been generated with predictions for various regions.

A folder `runs/model_name/latest` should also have been generated that contains copy of your model directory along with data files used. This can be useful to inspect if something goes wrong.


Step 3 (optional): Defining an environment for your model
.................................................

If needed, CHAP currently supports specifying a docker image or a python environment file that will be used when running your model.

We implement the MLProject standard, as described in the `MLflow documentation <https://www.mlflow.org/docs/latest/projects.html#project-format>`_ (except for conda support). Specifying a Python environment requires that you have pyenv installed and available.

