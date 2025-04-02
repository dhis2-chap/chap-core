# Running models through CHAP

## Running an external model on the command line

Models that are compatible with CHAP can be used with the `chap evaluate` command:

```bash
$ chap evaluate --model-name /path/to/your/model/directory --dataset-name ISIMIP_dengue_harmonized --dataset-country brazil --report-filename report.pdf --ignore-environment  --debug
```

Note the `--ignore-environment` in the above command. 
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

External models can be run on the command line using the `chap evaluate` command. See `chap evaluate --help` for details.

This example runs an auto ewars R model on public ISMIP data for Brazil using a public docker image with the R inla package. After running, a report file `report.pdf` should be made.


### Passing model-specific options to the model

We are currently working on experimental functionality for passing options and other parameters through Chap to the model.

The first way we plan to support this is when evaluating a model using the `chap evaluate` command.

This functionality is under development. Below is a minimal working example using the model `naive_python_model_with_mlproject_file_and_docker`. This model has a user_option `some_option`, which we can specify in a yaml file:

```bash
chap evaluate --model-name external_models/naive_python_model_with_mlproject_file_and_docker/ --dataset-name ISIMIP_dengue_harmonized --dataset-country vietnam --n-splits 2 --model-configuration-yaml external_models/naive_python_model_with_mlproject_file_and_docker/example_model_configuration.yaml
```

## Running an external model in Python

CHAP contains an API for loading models through Python. The following shows an example of loading and evaluating three different models by specifying paths/github urls, and evaluating those models:

```python
import pandas as pd

from chap_core.assessment.prediction_evaluator import evaluate_model
from chap_core.external.external_model import get_model_from_directory_or_github_url
from chap_core.file_io.file_paths import get_models_path
from chap_core.file_io.example_data_set import datasets
import logging

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    models_path = get_models_path()
    model_names = {
        #'deepar': models_path / 'deepar',
        'naive_model': models_path / 'naive_python_model_with_mlproject_file',
        # 'ewars': 'https://github.com/sandvelab/chap_auto_ewars'
    }

    dataset = datasets['ISIMIP_dengue_harmonized'].load()
    dataset = dataset['vietnam']
    n_tests = 7
    prediction_length = 6
    all_results = {}
    for name, model_name in model_names.items():
        model = get_model_from_directory_or_github_url(model_name)
        results = evaluate_model(model, dataset,
                                 prediction_length=prediction_length,
                                 n_test_sets=n_tests,
                                 report_filename=f'{name}_{n_tests}_{prediction_length}_report.pdf')
        all_results[name] = results

    report_file = 'evaluation_report.csv'
    df = pd.DataFrame([res[0] | {'model': name} for name, res in all_results.items()])
    df.to_csv(report_file, mode='w', header=True)

```