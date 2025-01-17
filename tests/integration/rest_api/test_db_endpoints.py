import json
import time

from pydantic import BaseModel

from chap_core.api_types import EvaluationEntry
from chap_core.database.base_tables import DBModel
from chap_core.database.database import SessionWrapper
from chap_core.database.debug import DebugEntry
from chap_core.database.tables import PredictionRead
from chap_core.rest_api_src.v1.rest_api import app
from fastapi.testclient import TestClient

from chap_core.rest_api_src.v1.routers.crud import BackTestFull, DatasetCreate, PredictionCreate
from chap_core.database.dataset_tables import DataSet, DataSetWithObservations

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
    path = f"/v1/crud/debug/{db_id}"
    response = client.get(path)
    data = DebugEntry.model_validate(response.json())
    assert data.timestamp > start_timestamp


def test_backtest_flow(celery_session_worker, clean_engine, dependency_overrides, weekly_full_data):
    with SessionWrapper(clean_engine) as session:
        dataset_id = session.add_dataset('full_data', weekly_full_data, 'polygons')
    response = client.post("/v1/crud/backtest",
                           json={"datasetId": dataset_id, "modelId": "naive_model"})
    assert response.status_code == 200, response.json()
    job_id = response.json()['id']
    db_id = await_result_id(job_id)
    response = client.get(f"/v1/crud/backtest/{db_id}")
    assert response.status_code == 200, response.json()
    BackTestFull.model_validate(response.json())
    response = client.get(f'/v1/analytics/evaluation-entry',
                          params={'backtestId': db_id, 'quantiles': [0.1, 0.5, 0.9]})
    assert response.status_code == 200, response.json()
    evaluation_entries = response.json()

    for entry in evaluation_entries:
        assert 'splitPeriod' in entry, f'splitPeriod not in entry: {entry.keys()}'
        EvaluationEntry.model_validate(entry)


def test_add_dataset_flow(celery_session_worker, dependency_overrides, dataset_create: DatasetCreate):
    response = client.post("/v1/crud/datasets", data=dataset_create.model_dump_json())
    assert response.status_code == 200, response.json()
    db_id = await_result_id(response.json()['id'])
    response = client.get(f"/v1/crud/datasets/{db_id}")
    assert response.status_code == 200, response.json()
    ds = DataSetWithObservations.model_validate(response.json())

    assert len(ds.observations) > 0
    print(response.json())
    assert 'orgUnit' in response.json()['observations'][0], response.json()['observations'][0].keys()


def test_make_prediction_flow(celery_session_worker, dependency_overrides, make_prediction_request):
    data = make_prediction_request.model_dump_json()
    response = client.post("/v1/analytics/prediction",
                           data=data)
    assert response.status_code == 200, response.json()
    db_id = await_result_id(response.json()['id'])
    response = client.get(f"/v1/crud/predictions/{db_id}")
    assert response.status_code == 200, response.json()
    ds = PredictionRead.model_validate(response.json())
    assert len(ds.forecasts) > 0
    assert all(len(f.values) for f in ds.forecasts)
    #assert 'orgUnit' in response.json()['observations'][0], response.json()['observations'][0].keys()


def test_add_csv_dataset(celery_session_worker, dependency_overrides, data_path):
    csv_data = open(data_path / 'nicaragua_weekly_data.csv', 'rb')
    geojson_data = open(data_path / 'nicaragua.json', 'rb')
    response = client.post('/v1/crud/datasets/csvFile', files={"csvFile": csv_data, "geojsonFile": geojson_data})
    assert response.status_code == 200, response.json()
