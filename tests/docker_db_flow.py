"""
This is meant to be a standalone python file for testing the flow with docker compose

This file should be possible to run without chap installed, only with the necessary requirements
"""

import requests
import json
import time
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def make_prediction_request(model_name):
    #ds = "../example_data/anonymous_chap_request.json"
    filename = '/home/knut/Data/ch_data/test_data/make_prediction_request.json'
    data = json.load(open(filename))
    data['modelId'] = model_name
    print(data.keys())
    return data


hostname = 'chap'
chap_url = "http://%s:8000" % hostname


class IntegrationTest:
    def __init__(self, chap_url, run_all):
        self._chap_url = chap_url
        self._run_all = run_all

    def main(self):
        ensure_up(self._chap_url)
        model_url = self._chap_url + "/v1/crud/models"
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
        assert 'naive_model' in {model['name'] for model in model_list}

        if self._run_all:
            for model in model_list:
                make_prediction(chap_url, make_prediction_request(model['name']))
        else:
            make_prediction(chap_url, make_prediction_request('naive_model'))


def wait_for_db_id(chap_url, job_id):
    ...


def make_prediction(chap_url, data):
    make_prediction_url = chap_url + "/v1/analytics/prediction"
    response = requests.post(make_prediction_url, json=data)
    assert response.status_code == 200, (response, response.text)
    return response.json()['id']


def ensure_up(chap_url):
    response = None
    for _ in range(5):
        try:
            response = requests.get(chap_url + "/v1/health")
            break
        except requests.exceptions.ConnectionError as e:
            logger.error("Failed to connect to %s" % chap_url)
            logger.error(e)
            time.sleep(5)
    assert response is not None
    assert response.status_code == 200, response.status_code
    assert response.json()["status"] == "success"


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
    results = requests.get(chap_url + "/v1/get-results")

    if results.status_code != 200:
        exception_info = requests.get(chap_url + "/v1/get-exception").json()
        logger.error("Failed to get results")
        logger.error(exception_info)
        raise ValueError("Failed to get results")

    assert results.status_code == 200, results.status_code
    print("STatus code", results.status_code)
    print("RESULTS", results)
    results = results.json()
    assert len(results['dataValues']) == 45, len(results['dataValues'])


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        hostname = sys.argv[1]
    else:
        hostname = 'localhost'
    chap_url = "http://%s:8000" % hostname
    IntegrationTest(chap_url, False).main()
