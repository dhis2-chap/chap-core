import json

from pydantic.v1.json import pydantic_encoder

from chap_core.assessment.forecast import forecast_ahead
from chap_core.assessment.prediction_evaluator import backtest
from chap_core.climate_predictor import QuickForecastFetcher
from chap_core.datatypes import FullData
from chap_core.predictor.model_registry import registry
from chap_core.rest_api_src.worker_functions import sample_dataset_to_prediction_response, \
    samples_to_evaluation_response, dataset_to_datalist
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

dataset = DataSet.from_csv('../../example_data/nicaragua_weekly_data.csv', FullData)
estimator = registry.get_model('auto_regressive_weekly')


predictions_list = backtest(estimator, dataset, prediction_length=12,
                           n_test_sets=20, stride=2, weather_provider=QuickForecastFetcher)

#predictions = forecast_ahead(estimator, dataset, 12)
response = samples_to_evaluation_response(
    predictions_list,
    quantiles=[0.05, 0.25, 0.5, 0.75, 0.95], real_data=dataset_to_datalist(dataset, 'dengue'))
#response = sample_dataset_to_prediction_response(predictions, 'dengue')
serialized_response = response.json()
out_filename = 'evaluation_response.json'
with open(out_filename, 'w') as out_file:
    out_file.write(serialized_response)