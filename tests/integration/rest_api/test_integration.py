
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
post_zip_file_path = "/v1/post-zip-file"


# Set the path to the model
#def test_post_set_model_path():
#    response = client.post(set_model_path_path, params={"model_path": "https://github.com/knutdrand/external_rmodel_example.git"})
#    assert response.status_code == 200

# Test get status on initial, should return 200
def test_post_zip_file():
    testfile = open("./testdata/traning_prediction_data.zip", "rb")
    print(testfile)
    response = client.post(post_zip_file_path, files={"file": testfile})
    print(response)
    #assert response.status_code == 200



# Test get status on initial, should return 200
def test_get_status():
    response = client.get(get_status_path)
    assert response.status_code == 200
    assert response.json()['ready'] == False
