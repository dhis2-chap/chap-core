import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from chap_core.api_types import BacktestParams
from chap_core.database.database import SessionWrapper
from chap_core.database.settings import get_settings
from chap_core.database.tables import BackendSettings, Backtest, BacktestSpecification
from chap_core.rest_api.app import app

client = TestClient(app)

METRIC_ORDER = [
    "crps_log1p",
    "winkler_score_25_75_log1p",
    "mae",
    "rmse",
    "mape",
    "coverage_10_90",
    "coverage_25_75",
    "crps",
    "crps_norm",
    "ratio_above_truth",
    "winkler_score_10_90",
    "winkler_score_10_90_log1p",
    "winkler_score_25_75",
]


def test_fresh_settings_are_seeded_once(override_session, seeded_session):
    assert seeded_session.get(BackendSettings, 1) is None
    response = client.get("/v1/settings")
    assert response.status_code == 200, response.text
    settings = response.json()
    assert settings["metricSortOrder"][:13] == METRIC_ORDER
    assert settings["selectedMetrics"] == METRIC_ORDER[:6]
    assert settings["defaultBaselineModel"] == "persistence"
    assert settings["monthlyBacktestParams"] == BacktestParams().model_dump(by_alias=True)
    assert settings["weeklyBacktestParams"] == BacktestParams(n_periods=12).model_dump(by_alias=True)
    assert client.get("/v1/settings").json() == settings
    assert len(seeded_session.exec(select(BackendSettings)).all()) == 1


def test_saved_settings_drive_leaderboard_and_survive_reseeding(
    override_session, p_seeded_engine, seeded_session, backend_settings_payload
):
    payload = backend_settings_payload
    all_metrics = set(payload["metricSortOrder"])
    # Partial order is allowed; unlisted metrics remain discoverable after this order.
    payload["metricSortOrder"] = ["coverage_10_90", "mae"]
    payload["selectedMetrics"] = ["mae", "coverage_10_90"]
    payload["defaultBaselineModel"] = "global_median"
    response = client.put("/v1/settings", json=payload)
    assert response.status_code == 200, response.text
    payload["selectedMetrics"] = ["coverage_10_90", "mae"]
    assert response.json() == payload
    # A new session (as on restart) uses the existing record, without restoring the seed.
    with Session(p_seeded_engine) as session:
        assert get_settings(session).model_dump(by_alias=True) == payload
    assert client.get("/v1/settings").json() == payload

    specification = seeded_session.exec(select(BacktestSpecification)).first()
    response = client.get(f"/v1/crud/backtest-specifications/{specification.id}")
    assert response.status_code == 200, response.text
    leaderboard = response.json()
    assert leaderboard["defaultBaselineModel"] == "global_median"
    metrics = leaderboard["metrics"]
    assert [metric["id"] for metric in metrics[:2]] == payload["metricSortOrder"]
    assert [metric["id"] for metric in metrics if metric["isSelected"]] == payload["selectedMetrics"]
    assert [metric["sortOrder"] for metric in metrics] == list(range(len(metrics)))
    assert {metric["id"] for metric in metrics} == all_metrics
    assert client.get("/v1/visualization/metrics/1").json() == metrics


@pytest.mark.parametrize(
    "field,value",
    [
        ("metricSortOrder", ["no_such_metric"]),
        ("metricSortOrder", ["mae", "mae"]),
        ("metricSortOrder", ["mae"]),
        ("selectedMetrics", ["no_such_metric"]),
        ("selectedMetrics", ["mae", "mae"]),
        ("defaultBaselineModel", "naive_model"),
        ("weeklyBacktestParams", {"nPeriods": 0}),
        ("monthlyBacktestParams", {"nSplits": 2, "nRetrain": 3}),
        ("monthlyBacktestParams", {"futureWeatherProvider": "no_such_provider"}),
    ],
)
def test_invalid_settings_do_not_replace_saved_settings(override_session, backend_settings_payload, field, value):
    original = client.get("/v1/settings").json()
    backend_settings_payload[field] = value
    response = client.put("/v1/settings", json=backend_settings_payload)
    assert response.status_code == 422, response.text
    assert client.get("/v1/settings").json() == original


@pytest.mark.parametrize(
    "period_type,settings_dataset_id", [("month", "month"), ("week", "week")], indirect=["settings_dataset_id"]
)
@pytest.mark.parametrize("endpoint", ["create-backtest", "create-backtests"])
def test_saved_defaults_are_used_by_backtests(
    override_session,
    p_seeded_engine,
    backend_settings_payload,
    analytics_worker,
    settings_dataset_id,
    period_type,
    endpoint,
):
    payload = backend_settings_payload
    payload["monthlyBacktestParams"].update(nPeriods=2, nSplits=3, stride=2, nRetrain=2)
    payload["weeklyBacktestParams"].update(nPeriods=4, nSplits=2, stride=3, nRetrain=1)
    assert client.put("/v1/settings", json=payload).status_code == 200
    expected = payload["weeklyBacktestParams" if period_type == "week" else "monthlyBacktestParams"]
    discovery = client.get("/v1/analytics/backtest-parameters", params={"periodType": period_type})
    assert discovery.status_code == 200, discovery.text
    assert {item["name"]: item["default"] for item in discovery.json()} == expected

    request = {"name": "saved defaults", "datasetId": settings_dataset_id}
    request.update({"modelId": "naive_model"} if endpoint == "create-backtest" else {"modelIds": ["naive_model"]})
    response = client.post(f"/v1/analytics/{endpoint}", json=request)
    assert response.status_code == 200, response.text
    func, args, kwargs = analytics_worker.queued[0]
    params = {name: kwargs[name] for name in BacktestParams.model_fields}
    assert BacktestParams(**params).model_dump(by_alias=True) == expected
    # Run the queued work and inspect the persisted specification, not just the mock arguments.
    with SessionWrapper(p_seeded_engine) as wrapper:
        backtest_id = func(*args, session=wrapper, **params)
        backtest = wrapper.session.get(Backtest, backtest_id)
        assert {name: getattr(backtest.specification, name) for name in params} == params
        if endpoint == "create-backtests":
            assert backtest.specification_id == response.json()["specificationId"]


@pytest.mark.parametrize("parameter", ["nRetrain", "n_retrain"])
def test_explicit_parameters_override_saved_defaults(
    override_session, backend_settings_payload, analytics_worker, seeded_session, parameter
):
    backend_settings_payload["monthlyBacktestParams"].update(nSplits=10, nRetrain=2)
    assert client.put("/v1/settings", json=backend_settings_payload).status_code == 200
    specification = seeded_session.exec(select(BacktestSpecification)).first()
    request = {"name": "override", "datasetId": specification.dataset_id, "modelId": "naive_model", parameter: 8}
    response = client.post("/v1/analytics/create-backtest", json=request)
    assert response.status_code == 200, response.text
    kwargs = analytics_worker.queued[0][2]
    assert (kwargs["n_splits"], kwargs["n_retrain"]) == (10, 8)

    request[parameter] = 11
    response = client.post("/v1/analytics/create-backtest", json=request)
    assert response.status_code == 422, response.text
    assert len(analytics_worker.queued) == 1


@pytest.mark.parametrize("weekly", [False, True])
def test_inline_backtest_uses_saved_defaults(
    override_session,
    backend_settings_payload,
    analytics_worker,
    dataset_make_request,
    dataset_make_request_weekly,
    weekly,
):
    payload = backend_settings_payload
    payload["monthlyBacktestParams"].update(nPeriods=2, nSplits=3, stride=2)
    payload["weeklyBacktestParams"].update(nPeriods=5, nSplits=4, stride=3)
    assert client.put("/v1/settings", json=payload).status_code == 200
    request = (dataset_make_request_weekly if weekly else dataset_make_request).model_dump(mode="json", by_alias=True)
    request["modelId"] = "naive_model"
    expected = payload["weeklyBacktestParams" if weekly else "monthlyBacktestParams"]

    response = client.post("/v1/analytics/create-backtest-with-data", json=request)
    assert response.status_code == 200, response.text
    params = analytics_worker.queued[0][2]["backtest_params"]
    assert BacktestParams(**params).model_dump(by_alias=True) == expected

    # Dry-run validation uses the same resolved parameters and never queues work.
    request.update(nSplits=1, nRetrain=2)
    response = client.post("/v1/analytics/create-backtest-with-data?dryRun=true", json=request)
    assert response.status_code == 422, response.text
    assert len(analytics_worker.queued) == 1


def test_no_metrics_selected_is_saved(override_session, backend_settings_payload):
    backend_settings_payload["selectedMetrics"] = []
    response = client.put("/v1/settings", json=backend_settings_payload)
    assert response.status_code == 200, response.text
    assert client.get("/v1/settings").json()["selectedMetrics"] == []


def test_backtest_parameter_discovery_rejects_unknown_period_type(override_session):
    assert client.get("/v1/analytics/backtest-parameters?periodType=day").status_code == 422
