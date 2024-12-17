import os
import time
import json
import pytest
import requests
from sqlmodel import Session

from chap_core.database.database import SessionWrapper
from chap_core.log_config import initialize_logging
from chap_core.rest_api_src.v1.rest_api import NaiveWorker, app
from fastapi.testclient import TestClient

from chap_core.rest_api_src.v1.routers.crud import BackTestFull
from chap_core.rest_api_src.v1.routers.dependencies import get_session
from chap_core.util import redis_available


#app.dependency_overrides[get_session] = get_test_session

client = TestClient(app)


def test_debug():
    response = client.post("/v1/crud/debug")
    assert response.status_code == 200
    assert response.json()['id']

@pytest.fixture
def dependency_overrides(clean_engine):
    def get_test_session():
        with Session(clean_engine) as session:
            yield session
    app.dependency_overrides[get_session] = get_test_session
    yield
    app.dependency_overrides.clear()


def await_result_id(job_id, timeout=30):
    for _ in range(timeout):
        response = client.get(f"/v1/jobs/{job_id}")
        status = response.json()
        if status == 'SUCCESS':
            return client.get(f"/v1/jobs/{job_id}/database_result").json()['id']
        if status == 'FAILURE':
            assert False, "Job failed"
        time.sleep(1)
    assert False, "Timed out"


def test_debug_flow(celery_session_worker, clean_engine, dependency_overrides):
    start_timestamp = time.time()
    response = client.post("/v1/crud/debug")
    assert response.status_code == 200
    job_id = response.json()['id']
    db_id = await_result_id(job_id)
    response = client.get(f"/v1/crud/debug/{db_id}")
    assert response.json()['timestamp'] > start_timestamp

def test_backtest_flow(celery_session_worker,clean_engine, dependency_overrides, weekly_full_data):
    with SessionWrapper(clean_engine) as session:
        dataset_id = session.add_dataset('full_data', weekly_full_data, 'polygons')
    start_timestamp = time.time()
    response = client.post("/v1/crud/backtest", json={"dataset_id": dataset_id, "estimator_id": "naive_model"})
    assert response.status_code == 200
    job_id = response.json()['id']
    db_id = await_result_id(job_id)
    response = client.get(f"/v1/crud/backtest/{db_id}")
    BackTestFull.model_validate(response.json())