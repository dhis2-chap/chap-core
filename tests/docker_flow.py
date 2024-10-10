"""
This is meant to be a standalone python file for testing the flow with docker compose

This file should be possible to run without chap installed, only with the necessary requirements
"""
import json
import time

import requests


def dataset():
    dataset = "../example_data/anonymous_chap_request.json"
    data = json.load(open(dataset))
    return data


#hostname = "localhost"
hostname = 'chap'
chap_url = "http://%s:8000" % hostname
#chap_with_r_inla_url = "http://localhost:8001"


def main():
    model_url = chap_url + "/v1/list-models"
    ensure_up(chap_url)
    models = requests.get(model_url)

    for model in models.json():
        evaluate_model(chap_url, dataset(), model)


def ensure_up(chap_url):
    for _ in range(5):
        try:
            requests.get(chap_url + "/v1/status")
            break
        except requests.exceptions.ConnectionError:
            time.sleep(5)


def evaluate_model(chap_url, data, model, timeout=120):
    ensure_up(chap_url)
    model_name = model["name"]
    data["estimator_id"] = model_name
    print(model_name)
    print(model)
    print("\n\n")
    evalute_url = chap_url + f"/v1/predict/"
    response = requests.post(evalute_url, json=data)
    assert response.status_code == 200
    assert response.json()["status"] == "success"

    for _ in range(timeout // 5):
        job_status = requests.get(chap_url + "/v1/status").json()
        print(job_status)
        time.sleep(5)
        if job_status['ready']:
            break
    else:
        raise TimeoutError("Model evaluation took too long")
    results = requests.get(chap_url + "/v1/get-results").json()
    assert len(results['dataValues']) == 45, len(results['dataValues'])

evaluate_model(chap_url, dataset(), {"name": "naive_model"})
