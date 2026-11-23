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


def test_the_principal_does_not_break_uvicorns_access_formatter():
    from uvicorn.logging import AccessFormatter

    args = ("127.0.0.1:5000", "GET", "/v1/jobs", "1.1", 200)
    record = logging.LogRecord("uvicorn.access", logging.INFO, "", 0, '%s - "%s %s HTTP/%s" %d', args, None)

    with principal_scope("token:caps"):
        PrincipalLogFilter().filter(record)

    assert "GET /v1/jobs" in AccessFormatter().format(record)


@pytest.mark.parametrize(
    "table", [Prediction, Backtest, DataSet, ConfiguredModelDB, PredictionSetup, Alert, AlertPolicy]
)
def test_every_owned_table_records_its_creator(table):
    assert "created_by" in table.__table__.columns


def test_only_the_creator_may_cancel_or_forget_a_job(
    client, celery_session_worker, clean_engine, dependency_overrides, monkeypatch, tmp_path, dataset_create
):
    config = tmp_path / "chap-server.yaml"
    config.write_text(
        "authentication:\n  tokens:\n"
        "    - {id: one, token_env: CHAP_TEST_ONE, tier: authenticated}\n"
        "    - {id: two, token_env: CHAP_TEST_TWO, tier: authenticated}\n"
    )
    monkeypatch.setenv("CHAP_CONFIG", str(config))
    monkeypatch.setenv("CHAP_TEST_ONE", "1" * 32)
    monkeypatch.setenv("CHAP_TEST_TWO", "2" * 32)
    monkeypatch.delenv("CHAP_API_TOKEN", raising=False)
    one, two = {"Authorization": "Bearer " + "1" * 32}, {"Authorization": "Bearer " + "2" * 32}

    job_id = client.post("/v1/crud/datasets", json=dataset_create.model_dump(mode="json"), headers=one).json()["id"]
    for _ in range(30):
        if client.get(f"/v1/jobs/{job_id}", headers=one).json()["status"] in ("SUCCESS", "FAILURE"):
            break
        time.sleep(1)

    assert client.post(f"/v1/jobs/{job_id}/cancel", headers=two).status_code == 404
    assert client.delete(f"/v1/jobs/{job_id}", headers=two).status_code == 404
    assert client.delete(f"/v1/jobs/{job_id}", headers=one).status_code == 200


def finished_record_id(client: TestClient, job_id: str, headers: dict) -> int:
    """Wait for a job made with ``headers`` and return the id of the record it stored."""
    for _ in range(60):
        status = client.get(f"/v1/jobs/{job_id}", headers=headers).json()["status"]
        if status in ("SUCCESS", "FAILURE"):
            break
        time.sleep(1)
    assert status == "SUCCESS", client.get(f"/v1/jobs/{job_id}/logs", headers=headers).text
    return client.get(f"/v1/jobs/{job_id}/database_result", headers=headers).json()["id"]


def test_a_prediction_setup_names_the_token(client, override_session, p_seeded_engine, seeded_session, caps_token):
    backtest = seeded_session.exec(select(Backtest)).first()
    assert backtest is not None
    payload = {"backtestId": backtest.id, "name": "Owned setup"}

    response = client.post("/v1/crud/prediction-setups", json=payload, headers=caps_token)

    assert response.status_code == 200, response.text
    assert created_by(p_seeded_engine, PredictionSetup, response.json()["id"]) == "token:caps"


def test_an_alert_names_the_token(client, clean_engine, dependency_overrides, caps_token):
    policy_id = client.post("/v1/crud/alert-policies", json=POLICY, headers=caps_token).json()["id"]
    alert = {"timePeriod": "2024-07", "orgUnit": "A", "alertPolicyId": policy_id, "level": "monitor"}

    response = client.post("/v1/crud/alerts", json={"alerts": [alert]}, headers=caps_token)

    assert response.status_code == 200, response.text
    (alert_id,) = response.json()["ids"]
    assert created_by(clean_engine, Alert, alert_id) == "token:caps"


def test_a_configured_model_names_the_token(client, clean_engine, dependency_overrides, caps_token):
    with Session(clean_engine) as session:
        seeded = session.exec(select(ConfiguredModelDB)).first()
        assert seeded is not None
        template_id = seeded.model_template_id

    response = client.post(
        "/v1/crud/configured-models", json={"name": "owned", "modelTemplateId": template_id}, headers=caps_token
    )

    assert response.status_code == 200, response.text
    assert created_by(clean_engine, ConfiguredModelDB, response.json()["id"]) == "token:caps"


def test_models_chap_seeds_itself_have_no_owner(tmp_path):
    from sqlalchemy import create_engine
    from sqlmodel import SQLModel

    from chap_core.database.model_template_seed import seed_configured_models_from_config_dir

    engine = create_engine(f"sqlite:///{tmp_path / 'seeded.db'}")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        seed_configured_models_from_config_dir(session, skip_chapkit_models=True)
        owners = {model.created_by for model in session.exec(select(ConfiguredModelDB))}

    assert owners == {None}


def test_a_backtest_the_worker_runs_names_the_token(
    client, celery_session_worker, clean_engine, dependency_overrides, caps_token, create_backtest_with_data_request
):
    payload = create_backtest_with_data_request.model_dump(mode="json")

    response = client.post("/v1/analytics/create-backtest-with-data/", json=payload, headers=caps_token)

    assert response.status_code == 200, response.text
    backtest_id = finished_record_id(client, response.json()["id"], caps_token)
    assert created_by(clean_engine, Backtest, backtest_id) == "token:caps"


def test_a_prediction_the_worker_makes_names_the_token(
    client, celery_session_worker, clean_engine, dependency_overrides, caps_token, dataset_make_request
):
    payload = dataset_make_request.model_dump(mode="json") | {"model_id": "naive_model", "n_periods": 3}

    response = client.post("/v1/analytics/make-prediction", json=payload, headers=caps_token)

    assert response.status_code == 200, response.text
    prediction_id = finished_record_id(client, response.json()["id"], caps_token)
    assert created_by(clean_engine, Prediction, prediction_id) == "token:caps"
