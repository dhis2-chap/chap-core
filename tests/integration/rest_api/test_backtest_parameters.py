from pydantic import Field
from starlette.testclient import TestClient

from chap_core.api_types import BacktestParams
from chap_core.assessment.weather_providers import DEFAULT_WEATHER_PROVIDER_ID
from chap_core.rest_api.app import app
from chap_core.rest_api.v1.routers import analytics


client = TestClient(app)


def test_backtest_parameters():
    response = client.get("/v1/analytics/backtest-parameters")
    assert response.status_code == 200, response.text
    parameters = response.json()
    defaults = BacktestParams().model_dump(by_alias=True)
    assert [parameter["name"] for parameter in parameters] == list(defaults)
    assert {parameter["name"]: parameter["default"] for parameter in parameters} == defaults

    by_name = {parameter["name"]: parameter for parameter in parameters}
    for name, label in {
        "nPeriods": "Forecast periods",
        "nSplits": "Number of splits",
        "stride": "Stride",
        "nRetrain": "Number of retrains",
    }.items():
        assert by_name[name]["label"] == label
        assert by_name[name]["type"] == "integer"
        assert by_name[name]["minimum"] == 1

    assert by_name["futureWeatherProvider"] == {
        "name": "futureWeatherProvider",
        "label": "Future-weather provider",
        "description": "Registered provider supplying climate covariates for each forecast window. "
        "See GET /v1/analytics/weather-providers.",
        "default": DEFAULT_WEATHER_PROVIDER_ID,
        "type": "string",
    }
    for field in BacktestParams.model_fields.values():
        assert by_name[field.alias]["description"] == field.description


def test_backtest_parameters_reflect_new_fields(monkeypatch):
    class ExtendedBacktestParams(BacktestParams):
        extra_periods: int = Field(default=4, ge=2, title="Extra periods", description="Extra forecast periods.")

    monkeypatch.setattr(analytics, "BacktestParams", ExtendedBacktestParams)
    response = client.get("/v1/analytics/backtest-parameters")
    assert response.status_code == 200, response.text
    assert response.json()[-1] == {
        "name": "extraPeriods",
        "label": "Extra periods",
        "description": "Extra forecast periods.",
        "default": 4,
        "type": "integer",
        "minimum": 2,
    }
