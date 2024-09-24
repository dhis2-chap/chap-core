import time
import json
import pytest

from climate_health.rest_api import app
from fastapi.testclient import TestClient

# main_backend()

client = TestClient(app)

# paths
set_model_path_path = "/v1/set-model-path"
get_status_path = "/v1/status"
post_zip_file_path = "/v1/zip-file"
list_models_path = "/v1/list-models"
list_features_path = "/v1/list-features"
get_result_path = "/v1/get-results"
predict_on_json_path = "/v1/predict-from-json"


# Set the path to the model
# def test_post_set_model_path():
#    response = client.post(set_model_path_path, params={"model_path": "https://github.com/knutdrand/external_rmodel_example.git"})
#    assert response.status_code == 200


# Test get status on initial, should return 200
# @pytest.mark.skip(reason="Waiting for background task to work")
@pytest.fixture(scope="session")
def rq_worker_process():
    # run 'rq worker' in a subprocess
    import subprocess

    process = subprocess.Popen(["rq", "worker"])
    yield process
    # get stdout and stderr from process
    process.terminate()
    # stdout, stderr = process.communicate()
    # print("----------------------")
    # print(stdout)
    # print("++++++++++++++++++++++")
    # print(stderr)

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
def test_predict_on_json_data(big_request_json, rq_worker_process):
    response = client.post(predict_on_json_path, json=json.loads(big_request_json))
    print(response, response.text[:100])
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


@pytest.mark.xfail(reason="Waiting for asyynch test client")
def test_get_status():
    response = client.get(get_status_path)
    assert response.status_code == 200
    assert response.json()["ready"] == False


def test_list_models():
    response = client.get(list_models_path)
    assert response.status_code == 200
    assert "HierarchicalStateModelD2" in {spec["name"] for spec in response.json()}


def test_list_features():
    response = client.get(list_features_path)
    assert response.status_code == 200
    assert {elem["id"] for elem in response.json()} == {
        "population",
        "rainfall",
        "mean_temperature",
    }
