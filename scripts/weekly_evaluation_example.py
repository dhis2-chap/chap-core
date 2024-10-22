from chap_core.assessment.prediction_evaluator import evaluate_model
from chap_core.climate_predictor import QuickForecastFetcher
from chap_core.data.datasets import ISIMIP_dengue_harmonized
from chap_core.datatypes import FullData
from chap_core.predictor.model_registry import registry
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

model = registry.get_model('chap_ewars_weekly')
dataset = DataSet.from_csv('example_data/nicaragua_weekly_data.csv', dataclass=FullData)
if __name__ == '__main__':
    evaluate_model(model,
                   dataset,
                   prediction_length=3,
                   n_test_sets=9,
                   report_filename='nicaragua_example_report.pdf',
                   weather_provider=QuickForecastFetcher)
