from chap_core.api_types import RequestV1


def test_validate_json(request_json):
    request = RequestV1.model_validate_json(request_json)
