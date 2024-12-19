import json
import os

import pytest
from ee import FeatureCollection
from pydantic_geojson import PointModel

from chap_core.api_types import RequestV1, DataList, DataElement, FeatureCollectionModel, FeatureModel
from chap_core.rest_api_src.worker_functions import train_on_json_data, predict, evaluate, get_combined_dataset
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet


@pytest.mark.skip("Missing valid data")
def test_train_on_json_data(request_json):
    train_on_json_data(request_json, "HierarchicalModel", "test_model_path")


@pytest.mark.skip("Missing valid data")
def test_train_on_json_data_big(big_request_json):
    train_on_json_data(big_request_json, "ProbabilisticFlaxModel", "test_model_path")


def test_train_on_json_data_new(big_request_json, mocked_gee):
    train_on_json_data(
        big_request_json, "https://github.com/sandvelab/chap_auto_ewars", None
    )


def test_predict(big_request_json, mocked_gee):
    predict(big_request_json)


def test_evaluate(big_request_json, mocked_gee):
    results = evaluate(big_request_json, n_splits=2, stride=1)
    print(results)


def test_dataset_from_request_v1(big_request_json, mocked_gee):
    from chap_core.rest_api_src.worker_functions import dataset_from_request_v1
    data = RequestV1.model_validate_json(big_request_json)
    dataset = dataset_from_request_v1(data)
    print(dataset)


@pytest.fixture
def combined_dataset():
    periods = ['202101', '202102']
    locations = ['location1', 'location2']
    data_elements = ['disease_cases', 'population', 'rainfall']
    data_lists = [
        DataList(dhis2Id='wierd'+data_element, featureId=data_element,
                 data=[DataElement(pe=period, value=i * j * k % 7, ou=location)
                       for i, period in enumerate(periods) for j, location in enumerate(locations)])
        for k, data_element in enumerate(data_elements)]
    geojson = FeatureCollectionModel(
        features=[FeatureModel(
            id=location, properties={"name": location},
            geometry=PointModel(coordinates=[0, 0]),
            type='Point')
                    for location in locations])

    return RequestV1(orgUnitsGeoJson=geojson, features=data_lists)


def test_get_combined_dataset(combined_dataset):
    dataset = get_combined_dataset(combined_dataset)
    assert isinstance(dataset, DataSet)
    for location, data in dataset.items():
        assert len(data) == 2
        assert len(data.rainfall) == 2
        assert len(data.population) == 2
