"""Failure bundles expose metadata without copying dataset values or requests."""

from types import SimpleNamespace

import pytest
from celery import Task
from fastapi.testclient import TestClient
from sqlalchemy import update
from sqlmodel import col, select

from chap_core.database.database import SessionWrapper
from chap_core.database.dataset_tables import DataSet, Observation
from chap_core.rest_api import celery_tasks
from chap_core.rest_api.app import app


client = TestClient(app)
path = "/v1/jobs/job-1/diagnostics"


def test_failed_job_diagnostics_contains_metadata_but_no_data_values(
    request_store, override_session, seeded_session, monkeypatch, tmp_path, org_units, seen_periods, feature_names
):
    monkeypatch.setattr(Task, "apply_async", lambda self, args, kwargs, **options: SimpleNamespace(id="job-1"))
    monkeypatch.setattr(celery_tasks, "CHAP_LOGS_DIR", tmp_path)
    monkeypatch.setattr(celery_tasks, "LOG_TAIL_BYTES", 40)
    (tmp_path / "task_job-1.debug.txt").write_text("first line is long enough to be cut\nsecond line\nlast line\n")
    model = SessionWrapper(session=seeded_session).get_configured_model_by_name("naive_model")
    model.model_template.source_url = "https://github.com/example/model"
    model.model_template.source_digest = "a" * 40
    dataset = seeded_session.exec(select(DataSet).where(DataSet.name == "dataset_with_nans")).one()
    sentinel = 987654321.125
    seeded_session.execute(
        update(Observation)
        .where(Observation.dataset_id == dataset.id, col(Observation.value).is_not(None))
        .values(value=sentinel)
    )
    seeded_session.commit()
    assert (
        client.post(
            "/v1/analytics/create-backtest",
            json={"name": "Failed backtest", "datasetId": dataset.id, "modelId": model.name, "nSplits": 4},
        ).status_code
        == 200
    )
    # Neither the original request nor arbitrary Redis fields belong in diagnostics.
    request_store.set("job_request:job-1", '{"providedData": [{"value": 987654321.125}]}')
    request_store.hset("job_meta:job-1", "unrelated", str(sentinel))
    celery_tasks.TrackedTask().on_failure(
        RuntimeError("Model failed"), "job-1", (), {}, SimpleNamespace(traceback="Traceback: Model failed")
    )

    response = client.get(path)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["job"]["name"] == "Failed backtest"
    assert body["job"]["status"] == "FAILURE"
    assert body["job"]["error"] == "Model failed"
    assert body["job"]["parameters"]["n_splits"] == 4
    assert body["job"]["parameters"]["n_periods"] == 3
    assert body["traceback"] == "Traceback: Model failed"
    assert body["model"] == {
        "name": model.name,
        "version": model.model_template.version,
        "source": "https://github.com/example/model",
        "commit_sha": "a" * 40,
    }
    assert body["dataset"] == {
        "n_locations": len(org_units),
        "period_type": "month",
        "first_period": seen_periods[0],
        "last_period": seen_periods[-1],
        "covariates": feature_names + ["disease_cases"],
        "missing_value_counts": {
            **dict.fromkeys(feature_names, 0),
            "disease_cases": len(org_units) * len(seen_periods) - 3,
        },
    }
    assert all(body["environment"].values())
    assert body["logs"] == {
        "tail": "[... earlier log output truncated ...]\nsecond line\nlast line\n",
        "truncated": True,
    }
    assert str(sentinel) not in response.text
    assert "providedData" not in response.text


def test_diagnostics_non_failed_job_returns_409(request_store, override_session):
    request_store.hset("job_meta:job-1", mapping={"status": "SUCCESS"})
    assert client.get(path).status_code == 409


def test_diagnostics_unknown_job_returns_404(request_store, override_session):
    assert client.get(path).status_code == 404


@pytest.mark.parametrize("references", [{}, {"model_id": "999999", "dataset_id": "999999"}])
def test_diagnostics_missing_metadata_and_logs(request_store, override_session, monkeypatch, tmp_path, references):
    monkeypatch.setattr(celery_tasks, "CHAP_LOGS_DIR", tmp_path)
    request_store.hset("job_meta:job-1", mapping={"status": "FAILURE", **references})

    response = client.get(path)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model"] is None
    assert body["dataset"] is None
    assert body["job"]["parameters"] == {}
    assert body["traceback"] is None
    assert body["logs"] == {"tail": "", "truncated": False}
