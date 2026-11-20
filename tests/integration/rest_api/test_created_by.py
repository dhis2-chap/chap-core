"""Records name the token that created them (CLIM-1965)."""

import logging
import time

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from chap_core.auth.context import PrincipalLogFilter, principal_scope
from chap_core.database.alert_tables import Alert, AlertPolicy
from chap_core.database.dataset_tables import DataSet
from chap_core.database.model_templates_and_config_tables import ConfiguredModelDB
from chap_core.database.tables import Backtest, Prediction, PredictionSetup
from chap_core.rest_api.app import app

TOKEN = "c" * 32
POLICY = {
    "name": "ladder",
    "levels": [
        {"name": "monitor", "thresholdParams": {"type": "seasonal", "stdMultiplier": 1.0}, "exceedanceThreshold": 0.3}
    ],
}


@pytest.fixture
def caps_token(monkeypatch, tmp_path):
    config = tmp_path / "chap-server.yaml"
    config.write_text("authentication:\n  tokens:\n    - {id: caps, token_env: CHAP_TEST_CAPS, tier: operator}\n")
    monkeypatch.setenv("CHAP_CONFIG", str(config))
    monkeypatch.setenv("CHAP_TEST_CAPS", TOKEN)
    monkeypatch.delenv("CHAP_API_TOKEN", raising=False)
    return {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def client():
    return TestClient(app)


def created_by(engine, table, record_id: int) -> str | None:
    with Session(engine) as session:
        record = session.get(table, record_id)
        assert record is not None
        return record.created_by


def test_a_record_names_the_token_that_created_it(client, clean_engine, dependency_overrides, caps_token):
    response = client.post("/v1/crud/alert-policies", json=POLICY, headers=caps_token)

    assert created_by(clean_engine, AlertPolicy, response.json()["id"]) == "token:caps"


def test_with_authentication_off_no_owner_is_recorded(client, clean_engine, dependency_overrides, monkeypatch):
    monkeypatch.delenv("CHAP_API_TOKEN", raising=False)
    monkeypatch.delenv("CHAP_CONFIG", raising=False)

    response = client.post("/v1/crud/alert-policies", json=POLICY)

    assert created_by(clean_engine, AlertPolicy, response.json()["id"]) is None


def test_a_job_and_what_it_creates_name_the_token(
    client, celery_session_worker, clean_engine, dependency_overrides, caps_token, dataset_create
):
    response = client.post("/v1/crud/datasets", json=dataset_create.model_dump(mode="json"), headers=caps_token)
    job_id = response.json()["id"]
    for _ in range(30):
        job = client.get(f"/v1/jobs/{job_id}", headers=caps_token).json()
        if job["status"] in ("SUCCESS", "FAILURE"):
            break
        time.sleep(1)

    assert job["status"] == "SUCCESS"
    assert job["created_by"] == "token:caps"
    dataset_id = client.get(f"/v1/jobs/{job_id}/database_result", headers=caps_token).json()["id"]
    assert created_by(clean_engine, DataSet, dataset_id) == "token:caps"


def test_the_request_log_names_the_principal():
    record = logging.LogRecord("uvicorn.access", logging.INFO, "", 0, '%s - "%s %s" %d', ("ip", "GET", "/", 200), None)

    with principal_scope("token:caps"):
        PrincipalLogFilter().filter(record)

    assert record.getMessage() == 'ip - "GET /" 200 token:caps'


@pytest.mark.parametrize(
    "table", [Prediction, Backtest, DataSet, ConfiguredModelDB, PredictionSetup, Alert, AlertPolicy]
)
def test_every_owned_table_records_its_creator(table):
    assert "created_by" in table.__table__.columns
