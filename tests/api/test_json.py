import pytest
from chap_core.api_types import RequestV1
from chap_core.dhis2_interface.pydantic_to_spatiotemporal import v1_conversion


@pytest.fixture
def request_model(request_json):
    return RequestV1.model_validate_json(request_json)


def test_validate_json(request_json):
    request = RequestV1.model_validate_json(request_json)
    # print(request)


def test_convert_pydantic(request_model):
    st = v1_conversion(request_model.features[0].data)
    print(st)
