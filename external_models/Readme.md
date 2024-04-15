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
