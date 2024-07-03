import time

import pytest

from unittest.mock import patch
from climate_health.rest_api import app
from fastapi.testclient import TestClient
from fastapi import FastAPI, Header, HTTPException
from fastapi import HTTPException
from fastapi.testclient import TestClient
from fastapi.testclient import TestClient
from climate_health.rest_api import main_backend

#main_backend()

client = TestClient(app)


#paths
set_model_path_path = "/v1/set-model-path"
get_status_path = "/v1/status"
post_zip_file_path = "/v1/zip-file"
list_models_path = "/v1/list-models"
list_features_path = "/v1/list-features"
get_result_path = "/v1/get-results"
# Set the path to the model
#def test_post_set_model_path():
#    response = client.post(set_model_path_path, params={"model_path": "https://github.com/knutdrand/external_rmodel_example.git"})
#    assert response.status_code == 200

# Test get status on initial, should return 200
#@pytest.mark.skip(reason="Waiting for background task to work")
@pytest.mark.asyncio
async def test_post_zip_file():
    testfile = open("./testdata/traning_prediction_data.zip", "rb")
    response = client.post(post_zip_file_path, files={"file": testfile})
    assert response.status_code == 200
    assert response.json()['status'] == "success"
    status = client.get(get_status_path)
    assert status.status_code == 200
    #assert status.json()['ready'] == False
    start_time = time.time()
    timeout = 30
    while client.get(get_status_path).json()['ready'] == False and time.time()-start_time < timeout:
        time.sleep(1)
    assert client.get(get_status_path).json()['ready'] == True
    result = client.get(get_result_path)
    assert result.status_code == 200
    assert 'diseaseId' in result.json()




# Test get status on initial, should return 200
@pytest.mark.xfail(reason="Waiting for asyynch test client")
def test_get_status():
    response = client.get(get_status_path)
    assert response.status_code == 200
    assert response.json()['ready'] == False

def test_list_models():
    response = client.get(list_models_path)
    assert response.status_code == 200
    assert 'HierarchicalStateModelD2' in {spec['name'] for spec in response.json()}

def test_list_features():
    response = client.get(list_features_path)
    assert response.status_code == 200
    assert {elem['id'] for elem in response.json()} == {'population', 'rainfall', 'mean_temperature'}
