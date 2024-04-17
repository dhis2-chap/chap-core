import json

import pandas as pd
import pytest

from climate_health.datatypes import HealthPopulationData
from climate_health.dhis2_interface.json_parsing import parse_disease_data, parse_population_data, join_data
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