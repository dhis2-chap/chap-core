from chap_core.models.external_web_model import ExternalWebModel


def test_model_information_is_none():
    model = ExternalWebModel(api_url="http://localhost:8000", name="test")
    assert model.model_information is None
