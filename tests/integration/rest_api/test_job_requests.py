"""Submitted requests and run metadata remain available after job failure."""

import json
from types import SimpleNamespace

import pytest
from celery import Task
from fastapi.testclient import TestClient
from sqlmodel import select

from chap_core.database.database import SessionWrapper
from chap_core.database.dataset_tables import DataSet
from chap_core.database.model_templates_and_config_tables import ConfiguredModelDB
from chap_core.database.tables import Backtest
from chap_core.rest_api import celery_tasks
from chap_core.rest_api.app import app
from chap_core.rest_api.celery_tasks import JOB_REQUEST_KW
from chap_core.rest_api.v1 import jobs


@pytest.mark.parametrize(
    "path",
    [
        "/v1/analytics/make-dataset",
        "/v1/analytics/create-backtest",
        "/v1/analytics/create-backtests",
        "/v1/analytics/create-backtest-with-data/",
        "/v1/analytics/make-prediction",
        "/v1/crud/datasets",
        "/v1/crud/prediction-setups/{setup_id}/run",
    ],
)
def test_submission_retains_request_and_metadata_after_failure(
    path, request_store, override_session, seeded_session, example_polygons, dataset_create, monkeypatch
):
    queued = []

    def dispatch(self, args, kwargs, **options):
        assert JOB_REQUEST_KW not in kwargs
        assert celery_tasks.JOB_METADATA_KW not in kwargs
        queued.append((args, kwargs))
        return SimpleNamespace(id=f"job-{len(queued)}")

    monkeypatch.setattr(Task, "apply_async", dispatch)
    payload = {
        "name": "Original request",
        "dataToBeFetched": [],
        "geojson": example_polygons.model_dump(mode="json"),
        "providedData": [
            {"featureName": feature, "period": f"{year}-{month:02d}", "orgUnit": polygon.id, "value": month}
            for feature in ["rainfall", "disease_cases", "population"]
            for year in range(2020, 2024)
            for month in range(1, 13)
            for polygon in example_polygons.features
        ],
    }
    model = SessionWrapper(session=seeded_session).get_configured_model_by_name("naive_model")
    payload["modelId"] = model.name
    if path == "/v1/analytics/create-backtest":
        payload = {
            "name": "Original request",
            "datasetId": seeded_session.exec(select(DataSet.id)).first(),
            "modelId": model.id,
        }
    elif path == "/v1/analytics/create-backtests":
        second_model = seeded_session.exec(select(ConfiguredModelDB).where(ConfiguredModelDB.id != model.id)).first()
        assert second_model is not None
        payload = {
            "name": "Original request",
            "datasetId": seeded_session.exec(select(DataSet.id)).first(),
            "modelIds": [model.name, second_model.id],
        }
    elif path == "/v1/crud/datasets":
        payload = dataset_create.model_dump(mode="json", by_alias=True, exclude_unset=True)
    elif "{setup_id}" in path:
        backtest = seeded_session.exec(select(Backtest)).first()
        setup_response = TestClient(app).post(
            "/v1/crud/prediction-setups",
            json={"backtestId": backtest.id, "name": "Request capture"},
        )
        assert setup_response.status_code == 200, setup_response.text
        setup_id = setup_response.json()["id"]
        path = path.format(setup_id=setup_id)
        payload.pop("dataToBeFetched", None)
        payload.pop("modelId")
        payload["nPeriods"] = 3
        payload["type"] = "forecasting"

    if "backtest" in path:
        payload.update(nSplits=4, stride=2)

    # Unknown fields and omitted defaults must survive exactly as submitted.
    if "prediction-setups" not in path:
        payload["clientContext"] = {"note": "reproduce æøå", "optional": None}
    response = TestClient(app).post(path, json=payload)
    assert response.status_code == 200, response.text

    assert len(queued) == (2 if path == "/v1/analytics/create-backtests" else 1)
    is_dataset = path in {"/v1/analytics/make-dataset", "/v1/crud/datasets"}
    expected_model = None if is_dataset else model
    if "prediction-setups" in path:
        expected_model = seeded_session.get(ConfiguredModelDB, backtest.model_db_id)
    for index, (args, kwargs) in enumerate(queued, start=1):
        if path == "/v1/analytics/create-backtests" and index == 2:
            expected_model = second_model
        expected_version = expected_model.model_template.version if expected_model else None
        job_id = f"job-{index}"
        metadata = celery_tasks.get_job_meta(job_id)
        assert metadata is not None
        assert metadata["status"] == "PENDING"
        assert "parameters" in metadata
        assert "provided_data" not in json.loads(metadata["parameters"])
        if not is_dataset:
            assert expected_model is not None
            assert int(metadata["model_id"]) == expected_model.id
            assert metadata["model_version"] == expected_version
            # Pin the worker to the same model version recorded in metadata.
            if "create-backtest" in path and "with-data" not in path:
                assert args[1].model_id == expected_model.id
            elif "with-data" in path:
                assert kwargs["model_id"] == expected_model.id
            else:
                assert kwargs["configured_model_id"] == expected_model.id
        celery_tasks.TrackedTask().on_failure(
            RuntimeError("model failed"), job_id, (), {}, SimpleNamespace(traceback="traceback")
        )
        job = next(job for job in TestClient(app).get("/v1/jobs").json() if job["id"] == job_id)
        assert job["status"] == "FAILURE"
        assert job["model_id"] == (expected_model.id if expected_model else None)
        assert job["model_version"] == expected_version
        assert job["dataset_id"] == payload.get("datasetId")
        assert job["parameters"] == json.loads(metadata["parameters"])
        if "backtest" in path:
            assert job["parameters"]["n_splits"] == 4
            assert job["parameters"]["stride"] == 2
            assert job["parameters"]["n_periods"] == 3
            assert job["parameters"]["n_retrain"] == 1
        elif not is_dataset:
            assert job["parameters"]["n_periods"] == 3
            assert job["parameters"]["future_weather_provider"] == "climatology"
        elif path.endswith("make-dataset"):
            assert job["parameters"]["type"] == "evaluation"
        else:
            assert job["parameters"] == {}
        download = TestClient(app).get(f"/v1/jobs/{job_id}/request")
        assert download.status_code == 200, download.text
        assert download.json() == payload
        assert 0 < request_store.ttl(f"job_request:{job_id}") <= celery_tasks.JOB_REQUEST_TTL_SECONDS


def test_missing_request_returns_404(request_store):
    assert TestClient(app).get("/v1/jobs/missing/request").status_code == 404


def test_delete_job_removes_request(request_store, monkeypatch):
    request_store.hset("job_meta:failed", mapping={"status": "FAILURE"})
    request_store.set("job_request:failed", '{"name":"original"}')
    monkeypatch.setattr(jobs.worker, "get_job", lambda _: SimpleNamespace(status="FAILURE"))
    client = TestClient(app)
    assert client.delete("/v1/jobs/failed").status_code == 200
    assert not request_store.exists("job_meta:failed", "job_request:failed")
    assert client.get("/v1/jobs/failed/request").status_code == 404


@pytest.mark.parametrize("body", [b"", b"hello"])
def test_non_json_body_is_rejected_by_validation(request_store, body):
    response = TestClient(app).post(
        "/v1/analytics/make-prediction", content=body, headers={"content-type": "text/plain"}
    )
    assert response.status_code == 422


def test_request_store_failure_does_not_fail_submission(request_store, override_session, seeded_session, monkeypatch):
    monkeypatch.setattr(Task, "apply_async", lambda self, args, kwargs, **options: SimpleNamespace(id="job-1"))

    def failing_set(*_args, **_kwargs):
        raise ConnectionError("redis down")

    monkeypatch.setattr(request_store, "set", failing_set)
    payload = {
        "name": "Original request",
        "datasetId": seeded_session.exec(select(DataSet.id)).first(),
        "modelId": "naive_model",
    }
    assert TestClient(app).post("/v1/analytics/create-backtest", json=payload).status_code == 200
    assert TestClient(app).get("/v1/jobs/job-1/request").status_code == 404
