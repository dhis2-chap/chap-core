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
