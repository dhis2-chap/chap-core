import json

from starlette.testclient import TestClient
from chap_core.rest_api_src.v1.rest_api import app
from integration.rest_api.test_db_endpoints import await_result_id

client = TestClient(app)


def test_make_prediction_request(celery_session_worker, clean_engine, dependency_overrides, local_data_path):
    request_data = open(local_data_path / 'predict_chap_request_data_2025-03-12T16_33_00.339Z.json', 'r').read()
    response = client.post("/v1/analytics/make-prediction", data=request_data)
    assert response.status_code == 200
    db_id = await_result_id(response.json()['id'], timeout=120)
    response = client.get(f"/v1/analytics/prediction-entry/{db_id}", params={'quantiles': [0.1, 0.5, 0.9]})
    assert response.status_code == 200, response.json()
    ds = [PredictionEntry.model_validate(entry) for entry in response.json()]
    assert len(ds) > 0
    assert all(pe.quantile in (0.1, 0.5, 0.9) for pe in ds)

