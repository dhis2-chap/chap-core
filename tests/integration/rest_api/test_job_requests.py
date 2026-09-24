"""Original API bodies are retained for download and removed with job metadata."""

from types import SimpleNamespace

import fakeredis
import pytest
from celery import Task
from fastapi.testclient import TestClient
from sqlmodel import select

from chap_core.database.dataset_tables import DataSet
from chap_core.database.tables import Backtest
from chap_core.rest_api import celery_tasks
from chap_core.rest_api.app import app
from chap_core.rest_api.celery_tasks import JOB_REQUEST_KW
from chap_core.rest_api.v1 import jobs
from chap_core.rest_api.v1.routers import crud


@pytest.fixture
def request_store(monkeypatch):
    store = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(celery_tasks, "r", store)
    monkeypatch.setattr(jobs, "redis", store)
    monkeypatch.setattr(crud, "redis", store)
    monkeypatch.delenv("CHAP_API_TOKEN", raising=False)
    return store


@pytest.mark.parametrize(
    "path",
    [
        "/v1/analytics/make-dataset",
        "/v1/analytics/create-backtest",
        "/v1/analytics/create-backtest-with-data/",
        "/v1/analytics/make-prediction",
        "/v1/crud/datasets",
        "/v1/crud/prediction-setups/{setup_id}/run",
    ],
)
def test_submission_retains_original_body(
    path, request_store, override_session, seeded_session, example_polygons, dataset_create, monkeypatch
):
    def dispatch(self, args, kwargs, **options):
        assert JOB_REQUEST_KW not in kwargs
        return SimpleNamespace(id="job-1")

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
    payload["modelId"] = "naive_model"
    if path == "/v1/analytics/create-backtest":
        payload = {
            "name": "Original request",
            "datasetId": seeded_session.exec(select(DataSet.id)).first(),
            "modelId": "naive_model",
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

    # Unknown fields and omitted defaults must survive exactly as submitted.
    if "prediction-setups" not in path:
        payload["clientContext"] = {"note": "reproduce æøå", "optional": None}
    response = TestClient(app).post(path, json=payload)
    assert response.status_code == 200, response.text

    download = TestClient(app).get("/v1/jobs/job-1/request")
    assert download.status_code == 200, download.text
    assert download.json() == payload
    assert 0 < request_store.ttl("job_request:job-1") <= celery_tasks.JOB_REQUEST_TTL_SECONDS


@pytest.mark.parametrize("metadata", [None, {"status": "FAILURE"}])
def test_missing_or_legacy_request_returns_404(request_store, metadata):
    if metadata:
        request_store.hset("job_meta:missing", mapping=metadata)
    response = TestClient(app).get("/v1/jobs/missing/request")
    assert response.status_code == 404


def test_delete_job_removes_request(request_store, monkeypatch):
    request_store.hset("job_meta:failed", mapping={"status": "FAILURE"})
    request_store.set("job_request:failed", '{"name":"original"}')
    monkeypatch.setattr(jobs.worker, "get_job", lambda _: SimpleNamespace(status="FAILURE"))
    client = TestClient(app)
    assert client.delete("/v1/jobs/failed").status_code == 200
    assert not request_store.exists("job_meta:failed", "job_request:failed")
    assert client.get("/v1/jobs/failed/request").status_code == 404
