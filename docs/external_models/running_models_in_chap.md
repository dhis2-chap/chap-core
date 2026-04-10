# Evaluating Models in Chap

In order to run Chap, you should first follow our [guide for how to install Chap](../chap-cli/chap-core-cli-setup.md).

Models that are compatible with Chap can be used with the `chap eval` command.
An external model can be provided to Chap in two ways:

- By specifying a path to a local code base:

```bash
$ chap eval --model-name /path/to/your/model/directory --dataset-csv https://raw.githubusercontent.com/dhis2/climate-health-data/refs/heads/main/lao/chap_LAO_admin1_monthly.csv --output-file eval.nc --plot --run-config.ignore-environment --run-config.debug
```

- By specifying a github URL to a git repo (the url needs to start with https://github.com/):

```bash
$ chap eval --model-name https://github.com/dhis2-chap/minimalist_example --dataset-csv https://raw.githubusercontent.com/dhis2/climate-health-data/refs/heads/main/lao/chap_LAO_admin1_monthly.csv --output-file eval.nc --plot --run-config.ignore-environment --run-config.debug
```

Note the `--run-config.ignore-environment` in the above commands.
This means that we don't ask Chap to use Docker or a Python environment when running the model.
This can be useful when developing and testing custom models before deploying them to a production environment.
Instead the model will be run directly using the current environment you are in.
This usually works fine when developing a model, but requires you to have both chap-core and the dependencies of your model available.

As an example, the following command runs the chap_auto_ewars model (this does not use --run-config.ignore-environment and will set up
a docker container based on the specifications in the MLproject file of the model):

```bash
$ chap eval --model-name https://github.com/dhis2-chap/chap_auto_ewars --dataset-csv https://raw.githubusercontent.com/dhis2/climate-health-data/refs/heads/main/lao/chap_LAO_admin1_monthly.csv --output-file eval.nc --plot
```

If the above command runs without any error messages, you have successfully evaluated the model through Chap, and an `eval.nc` file should have been generated along with an HTML plot with predictions for various regions.

A folder `runs/model_name/latest` should also have been generated that contains copy of your model directory along with data files used. This can be useful to inspect if something goes wrong.

### (Experimental:) Passing model-specific options to the model

We are currently working on experimental functionality for passing options and other parameters through Chap to the model.

The first way we plan to support this is when evaluating a model using the same `chap eval` command as described above.

This functionality is under development. Below is a minimal working example using the model `naive_python_model_with_mlproject_file_and_docker`. This model has a user_option `some_option`, which we can specify in a yaml file:

```bash
chap eval --model-name external_models/naive_python_model_with_mlproject_file_and_docker/ --dataset-csv https://raw.githubusercontent.com/dhis2/climate-health-data/refs/heads/main/lao/chap_LAO_admin1_monthly.csv --output-file eval.nc --plot --backtest-params.n-splits 2 --model-configuration-yaml external_models/naive_python_model_with_mlproject_file_and_docker/example_model_configuration.yaml
```
