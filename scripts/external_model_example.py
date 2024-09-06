from climate_health.assessment.prediction_evaluator import evaluate_model
from climate_health.external.external_model import get_model_from_directory_or_github_url
from climate_health.file_io.example_data_set import datasets

model_name = 'external_models/naive_python_model_with_mlproject_file'
model = get_model_from_directory_or_github_url(model_name)

dataset = datasets['ISIMIP_dengue_harmonized'].load()
dataset = dataset['brazil']

results = evaluate_model(model, dataset, prediction_length=3, n_test_sets=4, report_filename='report.pdf')

print(results)
