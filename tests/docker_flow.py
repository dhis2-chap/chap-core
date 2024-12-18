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


hostname = 'chap'
chap_url = "http://%s:8000" % hostname


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

    model_list = models.json()

    for model in model_list:
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
        response = requests.get(chap_url + "/v1/status")
        if response.status_code != 200:
            logger.error("Failed to get status")
            logger.error(response)
            continue
        job_status = response.json()
        if job_status['status'] == "failed":
            exception_info = requests.get(chap_url + "/v1/get-exception").json()
            if "Earth Engine client library not initialized" in exception_info:
                logger.warning("Exception: %s" % exception_info)
                raise Exception("Failed, Earth Engine client library not initialized")

            raise ValueError("Model evaluation failed. Exception: %s" % exception_info)

        logger.info(job_status)

        print("Waiting for model to finish")
        time.sleep(5)
        if job_status['ready']:
            break
    else:
        raise TimeoutError("Model evaluation took too long")
    results = requests.get(chap_url + "/v1/get-results").json()
    assert len(results['dataValues']) == 45, len(results['dataValues'])


if __name__ == "__main__":
    main()
