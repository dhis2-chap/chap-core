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


chap_url = "http://localhost:8000"
chap_with_r_inla_url = "http://localhost:8001"


def main():

    modellist = chap_url + "/v1/list-models"
    models = requests.get(modellist)

    for model in models.json():
        evaluate_model(chap_url, dataset(), model)


def evaluate_model(chap_url, data, model):
    model_name = model["name"]
    data["estimator_id"] = model_name
    print(model_name)
    print(model)
    print("\n\n")
    evalute_url = chap_url + f"/v1/predict/"
    response = requests.post(evalute_url, json=data)
    assert response.status_code == 200
    assert response.json()["status"] == "success"

    while True:
        job_status = requests.get(chap_url + "/v1/status").json()
        print(job_status)
        time.sleep(1)

#main()
evaluate_model(chap_url, dataset(), {"name": "naive_model"})
