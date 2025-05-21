"""
This is meant to be a standalone python file for testing the flow of the modelling app
This file should not import chap as a python package
# This file should be possible to run without chap pip installed, only with the necessary requirements
"""

import requests
import json
import time
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def make_prediction_request(model_name):
    filename = '../example_data/anonymous_make_prediction_request.json'
    data = json.load(open(filename))
    data['modelId'] = model_name
    print(data.keys())
    return data


def make_dataset_request():
    #filename = '/home/knut/Downloads/request_make_dataset.json'
    # filename = '/home/knut/Data/ch_data/test_data/make_dataset_request.json'
    #filename = '/home/knut/Downloads/new_dataset_chap_request_data_2025-05-12T14_45_58.309Z.json'
    filename = '../example_data/anonymous_make_dataset_request.json'
    data = json.load(open(filename))
    return data


def make_dataset_request2():
    filename = '../example_data/anonymous_make_dataset_request2.json'
    data = json.load(open(filename))
    return data


hostname = 'chap'
chap_url = "http://%s:8000" % hostname


class IntegrationTest:
    def __init__(self, chap_url, run_all):
        self._chap_url = chap_url
        self._run_all = run_all
        self._default_model = 'chap_ewars_monthly'

    def ensure_up(self):
        response = None
        logger.info("Ensuring %s is up" % self._chap_url)
        for _ in range(20):
            try:
                response = requests.get(self._chap_url + "/v1/health")
                break
            except requests.exceptions.ConnectionError as e:
                logger.error("Failed to connect to %s" % self._chap_url)
                logger.error(e)
                time.sleep(5)
        assert response is not None
        assert response.status_code == 200, response.status_code
        assert response.json()["status"] == "success"

    def _get(self, url):
        try:
            response = requests.get(url)
        except:
            logger.error("Failed to connect to %s" % chap_url)
            logger.error('Failed to get %s' % url)
            exception_info = requests.get(chap_url + "/v1/get-exception").json()
            logger.error(exception_info)
            raise
        assert response.status_code == 200, (response.status_code, response.text)
        return response.json()

    def _post(self, url, json):
        try:
            response = requests.post(url, json=json)
        except:
            logger.error("Failed to connect to %s" % chap_url)
            logger.error('Failed to get %s' % url)
            raise
        assert response.status_code == 200, (response.status_code, response.text)
        return response.json()

    def get_models(self):
        model_url = self._chap_url + "/v1/crud/models"
        models = self._get(model_url)
        # hacky remove autoreg weekly
        # TODO: delete after new version is published
        models = [model for model in models if model['name'] != 'auto_regressive_weekly']
        return models

    def make_prediction(self, data):
        logger.info(f'Making prediction for {data["modelId"]}')
        make_prediction_url = self._chap_url + "/v1/analytics/make-prediction"
        response = self._post(make_prediction_url, json=data)
        job_id = response['id']
        db_id = self.wait_for_db_id(job_id)
        prediction_result = self._get(self._chap_url + f"/v1/crud/predictions/{db_id}")
        assert prediction_result['modelId'] == data['modelId']
        return prediction_result

    def prediction_flow(self):
        logger.info(f'Starting prediction flow tests')

        self.ensure_up()
        model_list = self.get_models()
        assert 'naive_model' in {model['name'] for model in model_list}
        if self._run_all:
            for model in model_list:
                self.make_prediction(make_prediction_request(model['name']))
        else:
            self.make_prediction(make_prediction_request(self._default_model))

    def evaluation_flow(self):
        logger.info(f'Starting evaluation flow tests')

        self.ensure_up()
        model_list = self.get_models()
        assert 'naive_model' in {model['name'] for model in model_list}

        data = make_dataset_request()
        #data = make_dataset_request2()
        dataset_id = self.make_dataset(data)

        if self._run_all:
            model_names = [model['name'] for model in model_list]
        else:
            model_names = [self._default_model]

        for model_name in model_names:
            result, backtest_id = self.evaluate_model(dataset_id, model_name)
            actual_cases = self._get(self._chap_url + f"/v1/analytics/actualCases/{backtest_id}")
            result_org_units = {e['orgUnit'] for e in result}
            org_units = {de['ou'] for de in actual_cases['data']}
            assert result_org_units == org_units, (result_org_units, org_units)

    def make_dataset(self, data):
        make_dataset_url = self._chap_url + "/v1/analytics/make-dataset"
        response = self._post(make_dataset_url, json=data)
        job_id = response['id']
        db_id = self.wait_for_db_id(job_id)
        return db_id

    def evaluate_model(self, dataset_id, model):
        logger.info(f'Making evaluation for {model}')
        job_id = self._post(self._chap_url + "/v1/crud/backtests/",
                            json={"modelId": model, "datasetId": dataset_id, 'name': 'integration_test'})['id']
        db_id = self.wait_for_db_id(job_id)
        evaluation_result = self._get(self._chap_url + f"/v1/crud/backtests/{db_id}")
        assert evaluation_result['modelId'] == model
        assert evaluation_result['datasetId'] == dataset_id
        assert evaluation_result['name'] == 'integration_test', evaluation_result
        assert evaluation_result['created'], evaluation_result['created']
        url_string = self._chap_url + f'/v1/analytics/evaluation-entry?backtestId={db_id}&quantiles=0.5'
        evaluation_entries = self._get(url_string)
        return evaluation_entries, db_id

    def wait_for_db_id(self, job_id):
        for _ in range(400):
            job_url = self._chap_url + f"/v1/jobs/{job_id}"
            job_status = self._get(job_url).lower()
            logger.info(job_status)
            if job_status == "failure":
                raise ValueError("Failed job")
            if job_status == "success":
                return self._get(job_url + "/database_result/")['id']
            time.sleep(1)
        raise TimeoutError("Job took too long")


if __name__ == "__main__":
    import argparse

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "0"):
            return False

    parser = argparse.ArgumentParser(description="Script to run docker db endpoint flows.")
    parser.add_argument("host", type=str, nargs='?', default="localhost", help="Chap REST server host. Defaults to localhost.")
    parser.add_argument("run_all", type=str2bool, nargs='?', default='true', help="Turn on to run tests for all available models (default), or turn off to test just a single model for quick local testing.")

    args = parser.parse_args()
    logger.info(args)

    chap_url = f"http://{args.host}:8000"
    suite = IntegrationTest(chap_url, args.run_all)
    suite.prediction_flow()
    suite.evaluation_flow()
