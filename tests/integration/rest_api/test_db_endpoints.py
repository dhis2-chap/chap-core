import time
import json
import pytest
from sqlmodel import Session

from chap_core.api_types import EvaluationEntry, RequestV1
from chap_core.database.database import SessionWrapper
from chap_core.rest_api_src.v1.rest_api import NaiveWorker, app
from fastapi.testclient import TestClient

from chap_core.rest_api_src.v1.routers.crud import BackTestFull, DataSet, DatasetCreate, DataSetWithObservations
from chap_core.rest_api_src.v1.routers.dependencies import get_session, get_database_url, get_settings
from chap_core.rest_api_src.worker_functions import WorkerConfig

# app.dependency_overrides[get_session] = get_test_session

client = TestClient(app)


def test_debug(celery_session_worker):
    response = client.post("/v1/crud/debug")
    assert response.status_code == 200
    assert response.json()['id']




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


def test_backtest_flow(celery_session_worker, clean_engine, dependency_overrides, weekly_full_data):
    with SessionWrapper(clean_engine) as session:
        dataset_id = session.add_dataset('full_data', weekly_full_data, 'polygons')
    response = client.post("/v1/crud/backtest", json={"dataset_id": dataset_id, "estimator_id": "naive_model"})
    assert response.status_code == 200
    job_id = response.json()['id']
    db_id = await_result_id(job_id)
    response = client.get(f"/v1/crud/backtest/{db_id}")
    BackTestFull.model_validate(response.json())
    evaluation_entries = client.get(f'/v1/analytics/evaluation_entry',
                                    params={'backtest_id': db_id, 'quantiles': [0.1, 0.5, 0.9]})
    for entry in evaluation_entries.json():
        EvaluationEntry.model_validate(entry)


def test_add_dataset_flow(celery_session_worker, dependency_overrides, dataset_create: DatasetCreate):
    response = client.post("/v1/crud/dataset/json", data=dataset_create.model_dump_json())
    assert response.status_code == 200
    db_id = await_result_id(response.json()['id'])
    response = client.get(f"/v1/crud/dataset/{db_id}")
    ds = DataSetWithObservations.model_validate(response.json())
    assert len(ds.observations) > 0


def test_add_csv_dataset(celery_session_worker, dependency_overrides, data_path):
    csv_data = open(data_path / 'nicaragua_weekly_data.csv', 'rb')
    geojson_data = open(data_path / 'nicaragua.json', 'rb')
    response = client.post('/v1/crud/dataset/csv_file', files={"csv_file": csv_data, "geojson_file": geojson_data})
    assert response.status_code == 200
