import sys

from chap_core.data.datasets import ISIMIP_dengue_harmonized
from chap_core.external.external_model import get_model_from_yaml_file
from chap_core.assessment.prediction_evaluator import evaluate_model
from chap_core.external.mlflow import ExternalModel

"""
model_name = 'config.yml'
#working_dir = '../external_models/ewars_Plus/'
working_dir = '../../chap_auto_ewars/'

#model = get_model_from_yaml_file(working_dir  + model_name, working_dir)

#model = ExternalMLflowModel(working_dir, working_dir="./")
model = ExternalMLflowModel("https://github.com/sandvelab/chap_auto_ewars", working_dir=working_dir)

dataset = ISIMIP_dengue_harmonized
for country, data in dataset.items():
    print(country)
    try:
        results, table = evaluate_model(data, model, max_splits=3, start_offset=24, return_table=True,
                                    callback=None,
                                    mode='prediction_summary',
                                        run_naive_predictor=False)
        print(results)
        print(table)
    except AssertionError as e:
        print(e)
        continue

"""
