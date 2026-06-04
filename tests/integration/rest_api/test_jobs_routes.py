import json
import logging
import time
import uuid

import pytest
from fastapi.testclient import TestClient

from chap_core.rest_api.app import app
from chap_core.rest_api.celery_tasks import JobDescription, JobType
from chap_core.rest_api.v1 import jobs
from chap_core.util import redis_available

logger = logging.getLogger(__name__)
client = TestClient(app)
base_path = "/v1/jobs"
evaluate_path = "/v1/evaluate"


class _StaticWorker:
    """Stand-in for CeleryPool that returns a fixed list — lets the route's filter
    logic be exercised without needing real Redis state."""

    def list_jobs(self):
        return [
            JobDescription(
                id="job-1",
                type=JobType.PREDICTION,
                name="Prediction one",
                status="STARTED",
                start_time="2026-01-01T00:00:00",
                end_time=None,
                result=None,
                prediction_setup_id=12,
            ),
            JobDescription(
                id="job-2",
                type=JobType.PREDICTION,
                name="Prediction two",
                status="STARTED",
                start_time="2026-01-01T00:00:01",
                end_time=None,
                result=None,
                prediction_setup_id=13,
            ),
        ]


def test_list_jobs_filters_by_prediction_setup_id(monkeypatch):
    monkeypatch.setattr(jobs, "worker", _StaticWorker())

    response = client.get(f"{base_path}?type=create_prediction&predictionSetupId=12")

    assert response.status_code == 200, response.json()
    data = response.json()
    assert [job["id"] for job in data] == ["job-1"]
    assert data[0]["prediction_setup_id"] == 12


@pytest.mark.skip(reason="Old API")
@pytest.mark.skipif(not redis_available(), reason="Redis not available")
@pytest.mark.celery(
    broker="redis://localhost:6379", backend="redis://localhost:6379", include=["chap_core.rest_api.celery_tasks"]
)
def test_predict(big_request_json, celery_session_worker, dependency_overrides):
    data = json.loads(big_request_json)
    data["estimator_id"] = "naive_model"
    # with unittest.mock.patch("chap_core.rest_api.worker_functions.Era5LandGoogleEarthEngine", gee_mock):
    response = client.post(evaluate_path, json=data)
    assert response.status_code == 200, response.json()
    # response = client.post(evaluate_path, json=data)
    print(response.json())
    task_id = response.json()["task_id"]

    for _ in range(200):
        status = client.get(f"{base_path}/{task_id}")
        assert status.status_code == 200, status.json()
        if status.json() not in ["PENDING", "STARTED"]:
            break
        logger.info(status.json())
        time.sleep(1)

    assert status.status_code == 200, status.json()
    assert status.json() == "SUCCESS"
    result = client.get(f"{base_path}/{task_id}/evaluation_result")
    assert result.status_code == 200, result.json()
    print(result.json())


# Job-id membership: routes that took a job_id used to either silently accept a
# bogus id (returning "PENDING" / empty logs / 200 cancel) or 500 with a leaked
# TaskRevokedError. Every such route should now 404 on unknown ids. A fresh uuid
# per test avoids picking up Redis state from any prior run that buggy code may
# have written.


@pytest.fixture
def unknown_job_id():
    return f"nonexistent-{uuid.uuid4()}"


@pytest.mark.skipif(not redis_available(), reason="Redis not available")
def test_get_job_status_unknown_id_returns_404(unknown_job_id):
    response = client.get(f"{base_path}/{unknown_job_id}")
    assert response.status_code == 404, response.text


@pytest.mark.skipif(not redis_available(), reason="Redis not available")
def test_cancel_job_unknown_id_returns_404(unknown_job_id):
    response = client.post(f"{base_path}/{unknown_job_id}/cancel")
    assert response.status_code == 404, response.text


@pytest.mark.skipif(not redis_available(), reason="Redis not available")
def test_get_logs_unknown_id_returns_404(unknown_job_id):
    response = client.get(f"{base_path}/{unknown_job_id}/logs")
    assert response.status_code == 404, response.text


@pytest.mark.skipif(not redis_available(), reason="Redis not available")
@pytest.mark.parametrize("suffix", ["database_result", "evaluation_result", "prediction_result"])
def test_result_endpoints_unknown_id_return_404(unknown_job_id, suffix):
    response = client.get(f"{base_path}/{unknown_job_id}/{suffix}")
    assert response.status_code == 404, response.text


class _FakeJob:
    def __init__(self, result):
        self.result = result
        self.status = "SUCCESS"
        self.is_finished = True


def _stub_successful_job(monkeypatch, result: int, job_id: str = "job-1"):
    """Make `/result` endpoints see job_id as an existing, successful job returning `result`."""
    from chap_core.rest_api.v1 import jobs as jobs_module

    monkeypatch.setattr(jobs_module, "get_job_meta", lambda _job_id: {"status": "success"})
    monkeypatch.setattr(jobs_module.worker, "get_job", lambda _job_id: _FakeJob(result))
    return job_id


def test_prediction_result_returns_prediction(monkeypatch, override_session):
    job_id = _stub_successful_job(monkeypatch, result=1)
    response = client.get(f"{base_path}/{job_id}/prediction_result")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["id"] == 1
    assert body["modelId"] == "naive_model"
    assert body["nPeriods"] == 3
    assert body["name"] == "test prediction"


def test_evaluation_result_returns_backtest(monkeypatch, override_session):
    job_id = _stub_successful_job(monkeypatch, result=1)
    response = client.get(f"{base_path}/{job_id}/evaluation_result")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["id"] == 1
    assert body["modelId"] == "naive_model"
    assert body["name"] == "test backtest"
    assert body["aggregateMetrics"] == {"MAE": 1.5}


def test_prediction_result_missing_row_returns_404(monkeypatch, override_session):
    job_id = _stub_successful_job(monkeypatch, result=999999)
    response = client.get(f"{base_path}/{job_id}/prediction_result")
    assert response.status_code == 404, response.text


def test_evaluation_result_missing_row_returns_404(monkeypatch, override_session):
    job_id = _stub_successful_job(monkeypatch, result=999999)
    response = client.get(f"{base_path}/{job_id}/evaluation_result")
    assert response.status_code == 404, response.text
