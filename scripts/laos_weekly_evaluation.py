from chap_core.assessment.prediction_evaluator import evaluate_model
from chap_core.climate_predictor import QuickForecastFetcher
from chap_core.datatypes import FullData
from chap_core.predictor.model_registry import registry
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

model_name = 'auto_regressive_weekly'
model = registry.get_model(model_name)
filename = 'weekly_laos_data'
dataset = DataSet.from_csv('~/Data/ch_data/%s.csv' % filename, dataclass=FullData)

if __name__ == '__main__':
    evaluate_model(model,
                   dataset,
                   prediction_length=12,
                   n_test_sets=41,
                   report_filename=f'{filename}_{model_name}_report.pdf',
                   weather_provider=QuickForecastFetcher)