## Integrating external models

An external model can be integrated into the code base by adding a directory for it inside this directory (i.e. inside `external_models/`).

The directory can contain necessary scripts and files for the model and needs to at least contain a config.yml file following this structure:

```yaml
name: [Name of the model]
train_command: [A template string for command that is run for training the model. Should contain {train_data} (which will be replaced with a train data file when .train() is called on the model) and {model} (whish will be replaced by a temp file name that the model is stored to).
setup_command: [Optional. If set, will be run as a command before training]
predict_command: [A template command for training. Should contain {future_data} (which will be replaced by a .csv file containing future data) and {model}.
conda: [Optional. Can point to a yml file specifying a conda environment that all the commands will be run through.]
```

Note that all paths are relative to the directory of the model.

An example:
```yaml
name: hydromet_dengue
train_command: "Rscript train.R {train_data} {model} map.graph"
setup_command: "Rscript setup.R"
predict_command: "Rscript predict.R {future_data} {model}"
conda: env.yml
```

A model can then be initiated using only the path of this yaml file:

```python
from climate_health.external.external_model import get_model_from_yaml_file
yaml_file = 'path/to/config.yml'
model = get_model_from_yaml_file(yaml)
model.setup()  # optinal, will run setup command if specified
model.train(train_data)
results = model.predict(future_climate_data)
```

### Testing
If you want to include the model in the automatic tests, add it to the list above the test `test_all_external_models_acceptance` in `tests/external/test_external_models.py`.



### Recommended approach for adding external R models

1: Make a local conda environment with R. Make a yaml file with this content and start from that:

```yaml
channels:
  - conda-forge
  - bioconda
  - defaults
dependencies:
  - r-essentials
  - r-base
  - conda-forge::r-fmesher
  - ca-certificates
  - certifi
  - openssl
```

If you store this file as 'env.yaml' you can make an environment with `conda env create --name NAME --file=env.yml`

2: Try to get the initial r script to run without crashing (does not need to give anything correct out)

  - The model might give you some data or instructions to give data on its format

3: Try to split the code into setup, train and predict that takes our data in our format as input and output

4: Make a conda env file from the final environment you have created (if you had to install anything else). Make sure things work if you create a new environment from that yml file. Note: You can create a yml file by dumping your current environment, but make sure you make a minimal file (google this).

5: Try to run it through the ExternalCommandLineModel class by following the instructions in the beginning of this document by following the instructions in the beginning of this document.
