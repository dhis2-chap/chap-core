import time
import json
import pytest

from chap_core.rest_api import app
from fastapi.testclient import TestClient

client = TestClient(app)

# paths
set_model_path_path = "/v1/set-model-path"
get_status_path = "/v1/status"
post_zip_file_path = "/v1/zip-file"
list_models_path = "/v1/list-models"
list_features_path = "/v1/list-features"
get_result_path = "/v1/get-results"
predict_on_json_path = "/v1/predict-from-json"
predict_path = "/v1/predict"


@pytest.fixture(scope="session")
def rq_worker_process():
    # run 'rq worker' in a subprocess
    import subprocess

    process = subprocess.Popen(["rq", "worker"])
    yield process
    # get stdout and stderr from process
    process.terminate()
    process.terminate()


@pytest.mark.asyncio
@pytest.mark.slow
@pytest.mark.skip
async def test_post_zip_file(tests_path, rq_worker_process):
    testfile = open(
        tests_path / "integration/rest_api/testdata/traning_prediction_data.zip", "rb"
    )
    response = client.post(post_zip_file_path, files={"file": testfile})
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    status = client.get(get_status_path)
    assert status.status_code == 200
    start_time = time.time()
    timeout = 30
    while (
            client.get(get_status_path).json()["ready"] == False
            and time.time() - start_time < timeout
    ):
        time.sleep(1)
    assert client.get(get_status_path).json()["ready"] == True
    result = client.get(get_result_path)
    assert result.status_code == 200
    assert "diseaseId" in result.json()


# @pytest.mark.asyncio
@pytest.mark.skip
def test_predict_on_json_data(big_request_json, rq_worker_process):
    endpoint_path = predict_on_json_path
    check_job_endpoint(big_request_json, endpoint_path)


def test_predict(big_request_json, rq_worker_process):
    check_job_endpoint(big_request_json, predict_path)


def check_job_endpoint(big_request_json, endpoint_path):
    response = client.post(endpoint_path, json=json.loads(big_request_json))
    assert response.status_code == 200
    status = client.get(get_status_path)
    assert status.status_code == 200
    start_time = time.time()
    timeout = 120
    while (
            client.get(get_status_path).json()["ready"] == False
            and time.time() - start_time < timeout
    ):
        time.sleep(1)
    assert client.get(get_status_path).json()["ready"]
    result = client.get(get_result_path)
    assert result.status_code == 200


@pytest.mark.xfail(reason="Waiting for asyynch test client")
def test_get_status():
    response = client.get(get_status_path)
    assert response.status_code == 200
    assert response.json()["ready"] == False


def test_list_models():
    response = client.get(list_models_path)
    assert response.status_code == 200
    assert "MultiRegionPoissonModel" in {spec["name"] for spec in response.json()}


def test_list_features():
    response = client.get(list_features_path)
    assert response.status_code == 200
    assert {elem["id"] for elem in response.json()} == {
        "population",
        "rainfall",
        "mean_temperature",
    }
