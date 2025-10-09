# Running models through Chap

In order to run Chap, you should first follow our [guide for how to install Chap](https://dhis2-chap.github.io/chap-core/chap-cli/chap-core-cli-setup.html).  

# Running models through the Chap command-line interface
Models that are compatible with CHAP can be used with the `chap evaluate` command.
An external model can be provided to CHAP in two ways:
- By specifying a path to a local code base:
```bash
$ chap evaluate --model-name /path/to/your/model/directory --dataset-name ISIMIP_dengue_harmonized --dataset-country brazil --report-filename report.pdf --ignore-environment  --debug
```
- By specifying a github URL to a git repo (the url needs to start with https://github.com/):
```bash
$ chap evaluate --model-name https://github.com/dhis2-chap/minimalist_example --dataset-name ISIMIP_dengue_harmonized --dataset-country brazil --report-filename report.pdf --ignore-environment  --debug
```

Note the `--ignore-environment` in the above commands. 
This means that we don't ask CHAP to use Docker or a Python environment when running the model. 
This can be useful when developing and testing custom models before deploying them to a production environment.
Instead the model will be run directly using the current environment you are in. 
This usually works fine when developing a model, but requires you to have both chap-core and the dependencies of your model available. 

As an example, the following command runs the chap_auto_ewars model on public ISMIP data for Brazil (this does not use --ignore-environment and will set up
a docker container based on the specifications in the MLproject file of the model):
```bash
$ chap evaluate --model-name https://github.com/dhis2-chap/chap_auto_ewars --dataset-name ISIMIP_dengue_harmonized --dataset-country brazil
```

If the above command runs without any error messages, you have successfully evaluated the model through CHAP, and a file `report.pdf` should have been generated with predictions for various regions.

A folder `runs/model_name/latest` should also have been generated that contains copy of your model directory along with data files used. This can be useful to inspect if something goes wrong.

### (Experimental:) Passing model-specific options to the model

We are currently working on experimental functionality for passing options and other parameters through Chap to the model.

The first way we plan to support this is when evaluating a model using the same `chap evaluate` command as described above.

This functionality is under development. Below is a minimal working example using the model `naive_python_model_with_mlproject_file_and_docker`. This model has a user_option `some_option`, which we can specify in a yaml file:

```bash
chap evaluate --model-name external_models/naive_python_model_with_mlproject_file_and_docker/ --dataset-name ISIMIP_dengue_harmonized --dataset-country vietnam --n-splits 2 --model-configuration-yaml external_models/naive_python_model_with_mlproject_file_and_docker/example_model_configuration.yaml
```
