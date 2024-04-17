import json

import pytest

from climate_health.dhis2_interface.json_parsing import parse_json
from climate_health.spatio_temporal_data.temporal_dataclass import SpatioTemporalDict


@pytest.fixture()
def json_data(data_path):
    return json.load(open(data_path/'dhis_response.json'))


def test_parse_json(json_data):
    parsed = parse_json(json_data)
    assert isinstance(parsed, SpatioTemporalDict)
