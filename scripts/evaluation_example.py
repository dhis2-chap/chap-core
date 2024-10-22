from chap_core.assessment.prediction_evaluator import evaluate_model
from chap_core.climate_predictor import QuickForecastFetcher
from chap_core.data.datasets import ISIMIP_dengue_harmonized
from chap_core.predictor.model_registry import registry
model = registry.get_model('chap_ewars_monthly')
dataset = ISIMIP_dengue_harmonized['vietnam']
if __name__ == '__main__':
    evaluate_model(model,
                   dataset,
                   prediction_length=3,
                   n_test_sets=9,
                   report_filename='vietnam_example_report.pdf',
                   weather_provider=QuickForecastFetcher)
