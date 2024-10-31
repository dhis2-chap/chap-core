import numpy as np
import pandas as pd

from chap_core.assessment.prediction_evaluator import evaluate_model, backtest
from chap_core.climate_predictor import QuickForecastFetcher
from chap_core.datatypes import FullData
from chap_core.predictor.model_registry import registry
from chap_core.rest_api_src.worker_functions import samples_to_evaluation_response, dataset_to_datalist
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

dataset = DataSet.from_csv('/home/knut/Data/ch_data/uganda_weekly_data_harmonized_human.csv', FullData)
model_name = 'auto_regressive_weekly'
estimator = registry.get_model(model_name)
predictions_list = backtest(estimator, dataset, prediction_length=12,
                            n_test_sets=20, stride=2, weather_provider=QuickForecastFetcher)
response = samples_to_evaluation_response(
    predictions_list,
    quantiles=[0.05, 0.25, 0.5, 0.75, 0.95],
    real_data=dataset_to_datalist(dataset, 'dengue'))

dataframe = pd.DataFrame([entry.dict() for entry in response.predictions])
dataframe.to_csv(f'uganda_weekly_evaluation_{model_name}.csv')

serialized_response = response.json()

out_filename = f'evaluation_response_uganda_{model_name}.json'

with open(out_filename, 'w') as out_file:
    out_file.write(serialized_response)

