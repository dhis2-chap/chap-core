import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from chap_core.api_types import BacktestParams
from chap_core.assessment.metrics import available_metrics
from chap_core.database.settings import SettingsManager, default_settings
from chap_core.database.tables import BackendSettings, BacktestSpecification
from chap_core.rest_api.app import app
from chap_core.rest_api.celery_tasks import JOB_METADATA_KW

client = TestClient(app)


def test_fresh_settings_are_code_defaults_and_not_stored(override_session, seeded_session):
    assert seeded_session.get(BackendSettings, 1) is None
    response = client.get("/v1/settings")
    assert response.status_code == 200, response.text
    settings = response.json()
    selected = ["crps_log1p", "winkler_score_25_75_log1p", "mae", "rmse", "mape", "coverage_10_90"]
    assert settings["metricSortOrder"][:6] == selected
    assert set(settings["metricSortOrder"]) == set(available_metrics)
    assert settings["selectedMetrics"] == selected
    assert settings["defaultBaselineModel"] == "persistence"
    assert settings["monthlyBacktestParams"] == BacktestParams().model_dump(by_alias=True)
    assert settings["weeklyBacktestParams"] == BacktestParams(n_periods=12, stride=4).model_dump(by_alias=True)
    assert seeded_session.get(BackendSettings, 1) is None


def test_saved_settings_store_only_changes_and_drive_leaderboard(
    override_session, p_seeded_engine, seeded_session, backend_settings_payload
):
    payload = backend_settings_payload
    # Partial order is allowed; unlisted metrics are appended in their default order.
    payload["metricSortOrder"] = ["coverage_10_90", "mae"]
    payload["selectedMetrics"] = ["mae", "coverage_10_90"]
    payload["defaultBaselineModel"] = "global_median"
    response = client.put("/v1/settings", json=payload)
    assert response.status_code == 200, response.text
    saved = response.json()
    assert saved["metricSortOrder"][:2] == ["coverage_10_90", "mae"]
    assert set(saved["metricSortOrder"]) == set(available_metrics)
    assert saved["selectedMetrics"] == ["coverage_10_90", "mae"]
    with Session(p_seeded_engine) as session:
        assert SettingsManager(session).get().model_dump(by_alias=True) == saved
        record = session.get(BackendSettings, 1)
    assert record is not None
    assert set(record.values) == {"metric_sort_order", "selected_metrics", "default_baseline_model"}

    specification = seeded_session.exec(select(BacktestSpecification)).first()
    response = client.get(f"/v1/crud/backtest-specifications/{specification.id}")
    assert response.status_code == 200, response.text
    leaderboard = response.json()
    assert leaderboard["defaultBaselineModel"] == "global_median"
    metrics = leaderboard["metrics"]
    assert [metric["id"] for metric in metrics] == saved["metricSortOrder"]
    assert [metric["id"] for metric in metrics if metric["isSelected"]] == saved["selectedMetrics"]
    assert [metric["sortOrder"] for metric in metrics] == list(range(len(metrics)))
    assert client.get("/v1/visualization/metrics/1").json() == metrics


def test_stored_settings_with_removed_metric_and_baseline_still_load(
    override_session, seeded_session, backend_settings_payload
):
    values = {
        "metric_sort_order": ["removed_metric", "mae"],
        "selected_metrics": ["removed_metric", "mae"],
        "default_baseline_model": "removed_baseline",
        "weekly_backtest_params": backend_settings_payload["weeklyBacktestParams"],
        "monthly_backtest_params": backend_settings_payload["monthlyBacktestParams"],
    }
    seeded_session.add(BackendSettings(values=values))
    seeded_session.commit()
    response = client.get("/v1/settings")
    assert response.status_code == 200, response.text
    settings = response.json()
    assert settings["metricSortOrder"][0] == "mae"
    assert "removed_metric" not in settings["metricSortOrder"]
    assert settings["selectedMetrics"] == ["mae"]
    assert settings["defaultBaselineModel"] == "persistence"


def test_otherwise_invalid_stored_settings_fall_back_to_defaults(
    override_session, seeded_session, backend_settings_payload
):
    values = default_settings().model_dump()
    values["weekly_backtest_params"]["future_weather_provider"] = "removed_provider"
    seeded_session.add(BackendSettings(values=values))
    seeded_session.commit()
    response = client.get("/v1/settings")
    assert response.status_code == 200, response.text
    assert response.json() == backend_settings_payload


@pytest.mark.parametrize(
    "field,value",
    [
        ("metricSortOrder", ["no_such_metric"]),
        ("defaultBaselineModel", "naive_model"),
    ],
)
def test_invalid_settings_do_not_replace_saved_settings(override_session, backend_settings_payload, field, value):
    original = client.get("/v1/settings").json()
    backend_settings_payload[field] = value
    response = client.put("/v1/settings", json=backend_settings_payload)
    assert response.status_code == 422, response.text
    assert client.get("/v1/settings").json() == original


@pytest.mark.parametrize("period_type,endpoint", [("month", "create-backtest"), ("week", "create-backtests")])
def test_saved_defaults_are_used_by_backtests(
    override_session,
    seeded_session,
    backend_settings_payload,
    analytics_worker,
    weekly_dataset_id,
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

    dataset_id = (
        weekly_dataset_id
        if period_type == "week"
        else seeded_session.exec(select(BacktestSpecification)).first().dataset_id
    )
    request = {"name": "saved defaults", "datasetId": dataset_id}
    request.update({"modelId": "naive_model"} if endpoint == "create-backtest" else {"modelIds": ["naive_model"]})
    response = client.post(f"/v1/analytics/{endpoint}", json=request)
    assert response.status_code == 200, response.text
    kwargs = analytics_worker.queued[0][2]
    params = {name: kwargs[name] for name in BacktestParams.model_fields}
    assert BacktestParams(**params).model_dump(by_alias=True) == expected
    assert kwargs[JOB_METADATA_KW]["parameters"] == {
        **BacktestParams(**params).model_dump(),
        "target_column": "disease_cases",
    }


def test_explicit_parameters_override_saved_defaults(
    override_session, backend_settings_payload, analytics_worker, seeded_session
):
    backend_settings_payload["monthlyBacktestParams"].update(nSplits=10, nRetrain=2)
    assert client.put("/v1/settings", json=backend_settings_payload).status_code == 200
    specification = seeded_session.exec(select(BacktestSpecification)).first()
    request = {"name": "override", "datasetId": specification.dataset_id, "modelId": "naive_model", "nRetrain": 8}
    response = client.post("/v1/analytics/create-backtest", json=request)
    assert response.status_code == 200, response.text
    kwargs = analytics_worker.queued[0][2]
    assert (kwargs["n_splits"], kwargs["n_retrain"]) == (10, 8)

    request["nRetrain"] = 11
    request["modelId"] = "no_such_model"
    response = client.post("/v1/analytics/create-backtest", json=request)
    assert response.status_code == 422, response.text
    assert response.json()["detail"][0]["loc"] == ["body"]
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
