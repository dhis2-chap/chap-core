# Describing your model in our yaml-based format

To make your model chap-compatible, you need your train and predict endpoints (as discussed [here](train_and_predict.md)) need to be formally defined in a [YAML format](https://en.wikipedia.org/wiki/YAML) that follows the popular [MLflow standard](https://www.mlflow.org/docs/latest/projects.html#project-format).
Your codebase need to contain a file named `MLproject` that defines the following:
- An entry point in the MLproject file called `train` with parameters `train_data` and `model`
- An entry point in the MLproject file called `predict` with parameters `historic_data`, `future_data`, `model` and `out_file`

These should contain commands that can be run to train a model and predict the future using that model. The model parameter should be used to save a model in the train step that can be read and used in the predict step. CHAP will provide all the data (the other parameters) when running a model.

Here is an [example of a valid MLproject file](https://github.com/dhis2-chap/minimalist_example/blob/main/MLproject) (taken from our minimalist_example).

The MLproject file can specify a docker image or Python virtual environment that will be used when running the commands. 
An example of this is the [MLproject file](https://github.com/dhis2-chap/minimalist_example_r/blob/main/MLproject) contained within our minimalist_example_r.

