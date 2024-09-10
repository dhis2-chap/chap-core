from climate_health.assessment.prediction_evaluator import evaluate_model
from climate_health.external.external_model import get_model_from_directory_or_github_url
from climate_health.external.r_models import models_path
from climate_health.file_io.example_data_set import datasets

model_names= {'deepar': models_path/ 'deepar',
              'naive_model': models_path/ 'naive_python_model_with_mlproject_file',
              'ewars': 'https://github.com/sandvelab/chap_auto_ewars'}

dataset = datasets['ISIMIP_dengue_harmonized'].load()
dataset = dataset['brazil']

all_results = {}
for name, model_name in model_names.items():
    model = get_model_from_directory_or_github_url(model_name)
    results = evaluate_model(model, dataset, prediction_length=6, n_test_sets=12, report_filename=f'{name}_report.pdf')
    all_results[name] = results

print(all_results)
