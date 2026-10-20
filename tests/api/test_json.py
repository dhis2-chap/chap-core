import pytest

from chap_core.api_types import RequestV1
from chap_core.rest_api.worker_functions import v1_conversion


@pytest.fixture
def request_model(request_json):
    return RequestV1.model_validate_json(request_json)


def test_validate_json(request_json):
    request = RequestV1.model_validate_json(request_json)


def test_convert_pydantic(request_model):
    data = request_model.features[0].data
    dataset = v1_conversion(data)
    assert set(dataset.locations()) == {d.ou for d in data}
