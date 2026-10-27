import datetime
from typing import Annotated, get_args

import pytest
from pydantic import BaseModel, TypeAdapter

import chap_core.rest_api.app  # noqa: F401  imports every API model
from chap_core.database.base_tables import DBModel, UtcDatetime
from chap_core.database.model_templates_and_config_tables import ModelTemplateInformation
from chap_core.rest_api.data_models import ModelTemplateRead


def test_dbmodel():
    class TestModel(DBModel):
        snake_case_name: str

    data = TestModel(snake_case_name="test")
    json_data = data.model_dump(by_alias=True)
    assert json_data == {"snakeCaseName": "test"}


@pytest.mark.parametrize(
    "value",
    [
        datetime.datetime(2026, 10, 19, 20, 10, 43),
        datetime.datetime(2026, 10, 19, 22, 10, 43, tzinfo=datetime.timezone(datetime.timedelta(hours=2))),
    ],
)
def test_utc_datetime_serializes_in_utc_with_an_offset(value):
    class TestModel(DBModel):
        created: UtcDatetime

    assert TestModel(created=value).model_dump(mode="json") == {"created": "2026-10-19T20:10:43+00:00"}


def _subclasses(cls):
    for subclass in cls.__subclasses__():
        yield subclass
        yield from _subclasses(subclass)


def _has_datetime(annotation) -> bool:
    return annotation is datetime.datetime or any(_has_datetime(arg) for arg in get_args(annotation))


def test_every_datetime_field_serializes_with_a_utc_offset():
    """A datetime field typed as plain `datetime` would reach clients without a zone."""
    models = {model for model in _subclasses(BaseModel) if model.__module__.startswith("chap_core")}
    checked = []
    for model in models:
        for name, field in model.model_fields.items():
            if not _has_datetime(field.annotation):
                continue
            annotation = Annotated[field.annotation, *field.metadata] if field.metadata else field.annotation
            serialized = TypeAdapter(annotation).dump_python(datetime.datetime(2026, 10, 19, 20, 10), mode="json")
            assert serialized == "2026-10-19T20:10:00+00:00", f"{model.__qualname__}.{name} must use UtcDatetime"
            checked.append(name)
    assert checked


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
