import os
import time
import json
import pytest
import requests

from chap_core.log_config import initialize_logging
from chap_core.rest_api_src.v1.rest_api import NaiveWorker, app
from fastapi.testclient import TestClient

from chap_core.util import redis_available

client = TestClient(app)

# paths
set_model_path_path = "/v1/set-model-path"
get_status_path = "/v1/status"
post_zip_file_path = "/v1/zip-file"
list_models_path = "/v1/list-models"
list_features_path = "/v1/list-features"
get_result_path = "/v1/get-results"
get_exception_info = "/v1/get-exception"
predict_on_json_path = "/v1/predict-from-json"
predict_path = "/v1/predict"
evaluate_path = "/v1/evaluate"
evaluation_result_path = "/v1/get-evaluation-results"


@pytest.fixture(scope="session")
def rq_worker_process():
    has_worker = os.environ.get("HAS_WORKER", False)
    if has_worker:
        yield None
    else:
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


@pytest.mark.skipif(not redis_available(), reason="Redis not available")
@pytest.mark.skip("This test is not working")
def test_evaluate(big_request_json, rq_worker_process, monkeypatch):
    check_job_endpoint(big_request_json, evaluate_path, evaluation_result_path)


@pytest.mark.skip(reason="Failing locally")
def test_evaluate_gives_correct_error_message(big_request_json, rq_worker_process, monkeypatch):
    # this test should fail since INLA does not exist. Check that we get a clean error message from the model propagated
    # all the way back to the exception info
    big_request_json = json.loads(big_request_json)
    big_request_json["estimator_id"] = "chap_ewars_monthly"
    big_request_json = json.dumps(big_request_json)
    monkeypatch.setattr("chap_core.rest_api_src.v1.rest_api.worker", NaiveWorker())
    #check_job_endpoint(big_request_json, evaluate_path, evaluation_result_path)
    exception_info = run_job_that_should_fail_and_get_exception_info(big_request_json, evaluate_path, evaluation_result_path)
    assert "there is no package called ‘INLA’" in exception_info.json() or "Rscript: not found" in exception_info.json(), exception_info.json()


@pytest.mark.skipif(not redis_available(), reason="Redis not available")
@pytest.mark.celery(broker="redis://localhost:6379",
                    backend="redis://localhost:6379",
                    include=['chap_core.rest_api_src.celery_tasks'])
def test_predict(big_request_json, celery_session_worker):
    check_job_endpoint(big_request_json, predict_path)

@pytest.mark.skipif(not redis_available(), reason="Redis not available")
@pytest.mark.celery(broker="redis://localhost:6379",
                    backend="redis://localhost:6379",
                    include=['chap_core.rest_api_src.celery_tasks'])
def test_evaluate(big_request_json, celery_session_worker):
    check_job_endpoint(big_request_json, evaluate_path, evaluation_result_path)

def test_model_that_does_not_exist(big_request_json, monkeypatch):
    # patch worker in rest_api to be NaiveWorker
    monkeypatch.setattr("chap_core.rest_api_src.v1.rest_api.worker", NaiveWorker())
    request_json = big_request_json
    request_json = json.loads(request_json)
    request_json["estimator_id"] = "does_not_exist"
    request_json = json.dumps(request_json)
    info = run_job_that_should_fail_and_get_exception_info(request_json, predict_path)
    assert "Unknown model id" in info.json()
    # todo: check that excpetion contains the correct error message


def run_job_that_should_fail_and_get_exception_info(big_request_json, endpoint_path, result_path=get_result_path):
    big_request_json = json.loads(big_request_json)
    big_request_json["model"] = "does_not_exist"
    big_request_json = json.dumps(big_request_json)
    status = run_job_and_get_status(big_request_json, endpoint_path)
    assert not status["ready"]  
    exception_info = client.get(get_exception_info)
    print(exception_info.json())
    return exception_info


def check_job_endpoint(big_request_json, endpoint_path, result_path=get_result_path):
    status = run_job_and_get_status(big_request_json, endpoint_path)
    assert status["ready"]
    print(result_path)
    result = client.get(result_path)
    assert result.status_code == 200
    exception_info = client.get(get_exception_info)
    assert exception_info.status_code == 200
    assert exception_info.json() == ""
    return status 


def run_job_and_get_status(big_request_json, endpoint_path):
    response = client.post(endpoint_path, json=json.loads(big_request_json))
    assert response.status_code == 200
    status = client.get(get_status_path)
    assert status.status_code == 200
    start_time = time.time()
    timeout = 120
    while (
            client.get(get_status_path).json()["ready"] is False
            and client.get(get_status_path).json()["status"] != "failed"
            and time.time() - start_time < timeout
    ):
        time.sleep(1)
    status = client.get(get_status_path).json()
    return status


@pytest.mark.xfail(reason="Waiting for asyynch test client")
def test_get_status():
    response = client.get(get_status_path)
    assert response.status_code == 200
    assert response.json()["ready"] == False



@pytest.mark.skipif(not redis_available(), reason="Redis not available")
def test_list_models():
    response = client.get(list_models_path)
    assert response.status_code == 200
    spec_names = {spec["name"] for spec in response.json()}
    assert "chap_ewars_monthly" in spec_names
    assert "chap_ewars_weekly" in spec_names
    spec = next(spec for spec in response.json() if spec["name"] == "chap_ewars_monthly")
    assert 'population' in (feature['id'] for feature in spec['features'])


@pytest.mark.skipif(not redis_available(), reason="Redis not available")
def test_list_features():
    response = client.get(list_features_path)
    assert response.status_code == 200
    assert {elem["id"] for elem in response.json()} == {
        "population",
        "rainfall",
        "mean_temperature",
    }


def set_model_in_json(json_str, model_id):
    data = json.loads(json_str)
    data["estimator_id"] = "naive_model"
    return json.dumps(data)


def test_run_job_with_too_little_data(big_request_json, monkeypatch):
    # predict-endpoint should give an error message
    # todo: change to a dataset with too little data
    # check that a correct message is returned and that status from predict
    # endpoint is failed
    # this test requires some functionality to go from json data to a dataset
    # that we can change and then go back (or to generate api data directory from a dataset)
    pass
    """"
    monkeypatch.setattr("chap_core.rest_api_src.v1.rest_api.worker", NaiveWorker())
    model_name = "naive_model" 
    data = set_model_in_json(big_request_json, model_name)
    client = TestClient(app)
    response = client.post(predict_path, json=json.loads(data))
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    """

