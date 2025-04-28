import json

import numpy as np

from chap_core.api_types import RequestV2
from chap_core.rest_api_src.data_models import DatasetMakeRequest
from chap_core.rest_api_src.v1.routers.analytics import MakePredictionRequest


def old():
    filepath = '/home/knut/Data/ch_data/chap_request.json'
    data: RequestV2 = RequestV2.model_validate_json(open(filepath, 'r').read())
    for features in data.features:
        for entry in features.data:
            entry.value = np.random.randint(1000, 2000)

    new_file_path = '../example_data/anonymous_chap_request.json'
    with open(new_file_path, 'w') as f:
        f.write(data.model_dump_json())


def anonymize_make_prediction_request():
    filename = '/home/knut/Data/ch_data/test_data/make_prediction_request.json'
    outfile = '/home/knut/Sources/climate_health/example_data/anonymous_make_prediction_request.json'
    anonymize_dataset(filename, outfile, model=MakePredictionRequest)


def anonymize_dataset(filename, outfile, model):
    request: model = model.model_validate_json(open(filename, 'r').read())
    feature_names = {observation.feature_name for observation in request.provided_data}
    print(feature_names)
    for observation in request.provided_data:
        if observation.feature_name == 'population':
            observation.value = 100000
        elif observation.feature_name == 'disease_cases':
            observation.value = np.random.randint(1000, 20000)
        else:
            raise
    with open(outfile, 'w') as f:
        json.dump(request.model_dump(), f, indent=2)


def anonymize_make_dataset_request():
    filename = '/home/knut/Data/ch_data/test_data/make_dataset_request.json'
    outfile = '/home/knut/Sources/climate_health/example_data/anonymous_make_dataset_request.json'
    anonymize_dataset(filename, outfile, model=DatasetMakeRequest)


if __name__ == '__main__':
    #anonymize_make_prediction_request()
    anonymize_make_dataset_request()
