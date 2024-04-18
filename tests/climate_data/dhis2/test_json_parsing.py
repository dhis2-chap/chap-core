import json

import numpy as np
import pandas as pd
import pytest

from climate_health.datatypes import HealthPopulationData
from climate_health.dhis2_interface.json_parsing import parse_disease_data, parse_population_data, join_data, predictions_to_json
from climate_health.spatio_temporal_data.temporal_dataclass import SpatioTemporalDict


@pytest.fixture()
def json_data(data_path):
    return json.load(open(data_path/'dhis_response.json'))

@pytest.fixture()
def population_data(data_path):
    return json.load(open(data_path/'population_Laos.json'))


@pytest.mark.skip(reason='This test is not yet implemented')
def test_parse_json(json_data):
    parsed = parse_disease_data(json_data)
    assert isinstance(parsed, pd.DataFrame)


def test_parse_population_data(population_data):
    lookup = parse_population_data(population_data)
    assert 'VWGSudnonm5' in lookup
    assert lookup['VWGSudnonm5'] == 272115


def test_join_population_and_health(population_data, json_data):
    joined = join_data(json_data, population_data)
    assert isinstance(joined, SpatioTemporalDict)
    for location, data in joined.items():
        assert isinstance(data.data(), HealthPopulationData)

def test_predictions_to_json():
    data = SpatioTemporalDict.from_pandas(pd.DataFrame({'time_period': ['2020-01', '2020-02'],
                                                         'disease_cases': [1, 2],
                                                         'population': [100, 200],
                                                         'location': ['loc1', 'loc2']}), HealthPopulationData)
    json_data = predictions_to_json(data, attribute_mapping={'disease_cases': 'D', 'population': 'P'})
    for element in json_data:
        assert element['dataElement'] in ['D', 'P']
        assert element['orgUnit'] in ['loc1', 'loc2']
        assert element['value'] in [1, 2, 100, 200, np.nan] or np.isnan(element['value'])
        assert element['period'] in ['2020-01', '2020-02']
