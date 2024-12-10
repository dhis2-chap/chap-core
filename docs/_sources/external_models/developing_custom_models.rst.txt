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

An easy way to get started is to clone our example barebone repository for a Python model, which can be found `here <https://github.com/dhis2-chap/minimalist_example>`_. This will give you a train.py and predict.py file that you can use as starting points, as well with an MLProject configuration file.

Step 1: Test/develop your model outside CHAP
.............................................

The following is a suggested workflow that can be used when developing and testing your model. For ease of development, we recommend a workflow where you can run your model without fully integrating it with CHAP first. This makes it easier to debug and test your model in isolation. You should still make sure your model handles the data formats that CHAP uses. The easiest way is to test directly on example data provided by CHAP. You can find such `example data here <https://github.com/dhis2-chap/chap-core/tree/dev/example_data/v0>`_.

Download the files from that directory, and test that you can train a model using the `training_data.csv`. You should write your trained model to file using the file name that is provided to your train method.

Here is an example of a train.py file that does a simple linear regression with the provided example data:

.. testcode::

    import pandas as pd
    from sklearn.linear_model import LinearRegression
    import joblib

    def train(csv_fn, model_fn):
        df = pd.read_csv(csv_fn)
        features = ['rainfall', 'mean_temperature']
        X = df[features]
        Y = df['disease_cases']
        Y = Y.fillna(0)  # set NaNs to zero (not a good solution, just for the example to work)
        model = LinearRegression()
        model.fit(X, Y)
        joblib.dump(model, model_fn)

    train('example_data/v0/training_data.csv', 'model.pkl')

.. testoutput:

Note that a model is written to file. Your predict code needs to take this model as input and use it to make predictions. The prediction entry point needs to take these files as input:

- A model file name
- A file with historic data (data before the prediction period)
- A file with future climate data (for the period we want to predict cases for)
- A file name that will be used when writing the predictions

The following shows an example of a prediction script. Note that we don't write the actual predictions to file, but we write samples that represent possible outcomes. How to sample predictions would depend on the model in use -- here we sample from a normal distribution based on the predicted outcomes.

.. testcode::

    def predict(model_fn, historic_data_fn, future_climatedata_fn, predictions_fn):
        df = pd.read_csv(future_climatedata_fn)
        cols = ['rainfall', 'mean_temperature']
        X = df[cols]
        model = joblib.load(model_fn)

        predictions = model.predict(X)

        train_data = pd.read_csv(historic_data_fn)
        y_train = train_data['disease_cases']
        X_train = train_data[cols]

        # Estimate the residual variance from the training data
        residuals = y_train - model.predict(X_train)
        residual_variance = np.var(residuals)

        # Generate sampled predictions by adding Gaussian noise
        n_samples = 20  # Number of samples you want
        sampled_predictions = []

        for i in range(n_samples):
            noise = np.random.normal(0, np.sqrt(residual_variance), size=predictions.shape)

            # add the samples to the dataframe we write as output
            df[f'sample_{i}'] = predictions + noise

        df.to_csv(predictions_fn, index=False)


    predict('model.pkl', 'example_data/v0/historic_data.csv', 'example_data/v0/future_data.csv', 'predictions.csv')


Make sure you are able to train your model and generate samples for predictions before moving on to the next step. Make sure you have output columns called `sample_0`, `sample_1`, etc. in your predictions file.


Step 2: Running your model through CHAP
.........................................

If your model is able to generate samples to a csv file as shown above, it should be fairly easy to run the model through the CHAP command line interface. Make sure you have `chap-core` installed before continuing.

The benefit of running your model through CHAP is that you can let CHAP handle all the data processing and evaluation, and you can easily compare your model to other models. To do this, you need to create an MLProject configuration file for your model. This file should specify the entry points for training and predicting, as well as any dependencies your model has. You can then run your model through CHAP using the CHAP CLI.

Here is an example of an MLProject configuration file for the example model above:

.. code-block::

    name: some_model_name

    adapters: {'disease_cases': 'disease_cases',
               'location': 'location',
               'time_period': 'time_period',
               'rainfall': 'rainfall',
               'mean_temperature': 'mean_temperature'}

    entry_points:
      train:
        parameters:
          train_data: path
          model: str
        command: "python train.py {train_data} {model}"
      predict:
        parameters:
          historic_data: path
          future_data: path
          model: str
          out_file: path
        command: "python predict.py {model} {historic_data} {future_data} {out_file}"

The important part here is that the entry points for train and predict give commands that work with your model. These will be run inside the directory containing your model code. If creating a train.py and predict.py like shown above, you need to make sure these can be run through the command line. See examples of how this can be done in the `minimalist_example repository <https://github.com/dhis2-chap/minimalist_example/blob/main/train.py>`_.

Place this MLProject file in the same directory as your model code. You can then run your model through CHAP using the following command:

.. code-block:: console

    $ chap evaluate --model-name /path/to/your/model/directory --dataset-name ISIMIP_dengue_harmonized --dataset-country brazil --report-filename report.pdf --ignore-environment  --debug

Note the `--ignore-environment`. This means that we don't ask CHAP to use Docker or a Python environment when running the model. Instead the model will be run directly using the current environment you are in. This usually works fine when developing a mode, but requires you to have both chap-core and the dependencies of your model available. The next step shows how to run your model in an isolated environment.

If the above command runs without any error messages, you have successfully evaluated your model through CHAP, and a file `report.pdf` should have been generated with predictions for various regions.

A folder `runs/model_name/latest` should also have been generated that contains copy of your model directory along with data files used. This can be useful to inspect if something goes wrong.


Step 3: Defining an environment for your model
.................................................

CHAP currently supports specifying a docker image or a python environment file that will be used when running your model.

We implement the MLProject standard, as described in the `MLflow documentation <https://www.mlflow.org/docs/latest/projects.html#project-format>`_ (except for conda support). Specifying a Python environment requires that you have pyenv installed and available.

