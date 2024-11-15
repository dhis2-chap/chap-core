"""
This is meant to be a standalone python file for testing the flow with docker compose

This file should be possible to run without chap installed, only with the necessary requirements
"""
import json
import time

import requests
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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
    try:
        models = requests.get(model_url)
    except:
        print("Failed to connect to %s" % chap_url)
        logger.error("Failed when fetching models")
        print("----------------Exception info----------------")
        exception_info = requests.get(chap_url + "/v1/get-exception").json()
        print(exception_info) 
        logger.error(exception_info)
        logger.error("Failed to connect to %s" % chap_url)
        raise

    for model in models.json():
        evaluate_model(chap_url, dataset(), model)


def ensure_up(chap_url):
    for _ in range(5):
        try:
            requests.get(chap_url + "/v1/status")
            break
        except requests.exceptions.ConnectionError:
            time.sleep(5)


def evaluate_model(chap_url, data, model, timeout=600):
    print("Evaluating", model)
    ensure_up(chap_url)
    model_name = model["name"]
    data["estimator_id"] = model_name
    logger.info(model_name)
    evalute_url = chap_url + f"/v1/predict/"
    response = requests.post(evalute_url, json=data)
    assert response.status_code == 200
    assert response.json()["status"] == "success"

    for _ in range(timeout // 5):
        job_status = requests.get(chap_url + "/v1/status").json()
        if job_status['status'] == "failed":
            exception_info = requests.get(chap_url + "/v1/get-exception").json()
            if "Earth Engine client library not initialized" in exception_info:
                logger.warning("Earth Engine client library not initialized. Ignoring this test")
                logger.warning("Exception: %s" % exception_info)
                return

            raise ValueError("Model evaluation failed. Exception: %s" % exception_info)

        logger.info(job_status)

        time.sleep(5)
        if job_status['ready']:
            break
    else:
        raise TimeoutError("Model evaluation took too long")
    results = requests.get(chap_url + "/v1/get-results").json()
    assert len(results['dataValues']) == 45, len(results['dataValues'])

if __name__ == "__main__":
    main()
    #

#evaluate_model(chap_url, dataset(), {"name": "naive_model"})
