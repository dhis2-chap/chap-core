import pytest

from chap_core.database.base_tables import DBModel
from chap_core.database.model_templates_and_config_tables import ModelTemplateInformation
from chap_core.rest_api.data_models import ModelTemplateRead


def test_dbmodel():
    class TestModel(DBModel):
        snake_case_name: str

    data = TestModel(snake_case_name="test")
    json_data = data.model_dump(by_alias=True)
    assert json_data == {"snakeCaseName": "test"}


@pytest.mark.parametrize(
    "min_key, max_key",
    [
        ("min_prediction_periods", "max_prediction_periods"),
        ("minPredictionPeriods", "maxPredictionPeriods"),
        ("min_prediction_length", "max_prediction_length"),
        ("minPredictionLength", "maxPredictionLength"),
    ],
)
def test_prediction_horizon_accepts_legacy_and_current_names(min_key, max_key):
    """MLproject files and API clients written against either spelling must parse."""
    info = ModelTemplateInformation.model_validate({min_key: 2, max_key: 6})

    assert info.min_prediction_periods == 2
    assert info.max_prediction_periods == 6


def test_model_template_read_serves_both_horizon_spellings():
    """The legacy camelCase keys stay in the response until clients have moved off them."""
    read = ModelTemplateRead.model_validate(
        {"name": "bounded_model", "id": 1, "minPredictionPeriods": 2, "maxPredictionPeriods": 6}
    )

    payload = read.model_dump(by_alias=True)

    assert payload["minPredictionPeriods"] == 2
    assert payload["maxPredictionPeriods"] == 6
    assert payload["minPredictionLength"] == 2
    assert payload["maxPredictionLength"] == 6
