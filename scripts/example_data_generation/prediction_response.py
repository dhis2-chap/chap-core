import json

from pydantic.v1.json import pydantic_encoder

from chap_core.assessment.forecast import forecast_ahead
from chap_core.datatypes import FullData
from chap_core.predictor.model_registry import registry
from chap_core.rest_api_src.worker_functions import sample_dataset_to_prediction_response
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

dataset = DataSet.from_csv('../../example_data/nicaragua_weekly_data.csv', FullData)
estimator = registry.get_model('auto_regressive_weekly')

predictions = forecast_ahead(estimator, dataset, 12)
response = sample_dataset_to_prediction_response(predictions, 'dengue')
serialized_response = json.dumps(response, default=pydantic_encoder)
out_filename = 'prediction_response.json'
with open(out_filename, 'w') as out_file:
    out_file.write(serialized_response)