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
    data['name'] = f'integration_test: {model_name}'
    return data


def make_dataset_request(dataset_path):
    # set default dataset path if empty
    if not dataset_path:
        dataset_path = '../example_data/anonymous_make_dataset_request.json' 
    # load dataset from json
    logger.info(f'Using dataset from file {dataset_path}')
    data = json.load(open(dataset_path))
    return data


def make_dataset_request2():
    filename = '../example_data/anonymous_make_dataset_request2.json'
    data = json.load(open(filename))
    return data


def make_backtest_with_data_request(model_name):
    # TODO: current file lacks population data so will only work with naive model
    filename = '../example_data/create-backtest-with-data.json'
    data = json.load(open(filename))
    data['modelId'] = model_name
    data['name'] = f'integration_test: {model_name} with data'
    return data


hostname = 'chap'
chap_url = "http://%s:8000" % hostname


class IntegrationTest:
    def __init__(self, chap_url, model_id=None, dataset_path=None):
        self._chap_url = chap_url
        self._model_id = model_id
        self._dataset_path = dataset_path

    def ensure_up(self):
        response = None
        logger.info("Ensuring %s is up" % self._chap_url)
        errors = []
        for _ in range(40):
            try:
                response = requests.get(self._chap_url + "/v1/health")
                break
            except requests.exceptions.ConnectionError as e:
                #logger.error("Failed to connect to %s" % self._chap_url)
                #logger.error(e)
                errors.append(e)
                time.sleep(5)
        else:
            logger.error("Failed to connect to %s after 40 attempts" % self._chap_url)
            for error in errors:
                logger.error(error)
            raise ConnectionError("Could not connect to %s" % self._chap_url)
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

    def _get_model_names(self, model_list):
        # hacky only consider monthly or any period models
        # TODO: delete after new version is published
        return [model['name'] for model in model_list if
                model['supportedPeriodType'] in ('month', 'any')]

    def make_dataset(self, data):
        make_dataset_url = self._chap_url + "/v1/analytics/make-dataset"
        response = self._post(make_dataset_url, json=data)
        job_id = response['id']
        db_id = self.wait_for_db_id(job_id)
        return db_id

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

        all_model_names = self._get_model_names(model_list)
        if self._model_id:
            assert self._model_id in all_model_names, f"Model {self._model_id} not found in {all_model_names}"
            model_names = [self._model_id]
        else:
            model_names = all_model_names

        errors = []
        for model_name in model_names:
            try:
                self.make_prediction(make_prediction_request(model_name))
            except Exception as err:
                msg = f'Error making prediction for model {model_name}: {err}'
                logger.error(msg)
                errors.append(msg)

        if errors:
            raise Exception('One or more prediction errors, for details see above logs')

    def evaluate_model(self, model, dataset_id):
        logger.info(f'Making evaluation for {model}')
        job_id = self._post(self._chap_url + "/v1/crud/backtests/",
                            json={"modelId": model, "datasetId": dataset_id, 'name': f'integration_test: {model}'})['id']
        db_id = self.wait_for_db_id(job_id)
        evaluation_result = self._get(self._chap_url + f"/v1/crud/backtests/{db_id}")
        assert evaluation_result['modelId'] == model
        assert evaluation_result['datasetId'] == dataset_id
        assert evaluation_result['name'].startswith('integration_test'), evaluation_result['name']
        assert evaluation_result['created'], evaluation_result['created']
        url_string = self._chap_url + f'/v1/analytics/evaluation-entry?backtestId={db_id}&quantiles=0.5'
        evaluation_entries = self._get(url_string)
        return evaluation_entries, db_id

    def evaluation_flow(self):
        logger.info(f'Starting evaluation flow tests')

        self.ensure_up()
        model_list = self.get_models()
        assert 'naive_model' in {model['name'] for model in model_list}

        data = make_dataset_request(self._dataset_path)
        dataset_id = self.make_dataset(data)

        all_model_names = self._get_model_names(model_list)
        if self._model_id:
            assert self._model_id in all_model_names, f"Model {self._model_id} not found in {all_model_names}"
            model_names = [self._model_id]
        else:
            model_names = all_model_names

        errors = []
        for model_name in model_names:
            try:
                result, backtest_id = self.evaluate_model(model_name, dataset_id)
                actual_cases = self._get(self._chap_url + f"/v1/analytics/actualCases/{backtest_id}")
                result_org_units = {e['orgUnit'] for e in result}
                org_units = {de['ou'] for de in actual_cases['data']}
                assert result_org_units == org_units, (result_org_units, org_units)
            except Exception as err:
                msg = f'Error making evaluation for model {model_name}: {err}'
                logger.error(msg)
                errors.append(msg)

        if errors:
            raise Exception('One or more evaluation errors, for details see above logs')

    def evaluate_model_with_data(self, data):
        logger.info(f'Making evaluation for {data["modelId"]}')
        job_id = self._post(self._chap_url + "/v1/analytics/create-backtest-with-data/", json=data)['id']
        db_id = self.wait_for_db_id(job_id)
        evaluation_result = self._get(self._chap_url + f"/v1/crud/backtests/{db_id}")
        assert evaluation_result['modelId'] == data["modelId"]
        assert evaluation_result['name'].startswith('integration_test'), evaluation_result['name']
        assert evaluation_result['created'], evaluation_result['created']
        url_string = self._chap_url + f'/v1/analytics/evaluation-entry?backtestId={db_id}&quantiles=0.5'
        evaluation_entries = self._get(url_string)
        return evaluation_entries, db_id

    def evaluation_with_data_flow(self):
        logger.info(f'Starting evaluation with data flow tests')

        self.ensure_up()
        model_list = self.get_models()
        assert 'naive_model' in {model['name'] for model in model_list}

        all_model_names = self._get_model_names(model_list)
        if self._model_id:
            assert self._model_id in all_model_names, f"Model {self._model_id} not found in {all_model_names}"
            model_names = [self._model_id]
        else:
            model_names = all_model_names

        errors = []
        for model_name in model_names:
            try:
                result, backtest_id = self.evaluate_model_with_data(make_backtest_with_data_request(model_name))
                actual_cases = self._get(self._chap_url + f"/v1/analytics/actualCases/{backtest_id}")
                result_org_units = {e['orgUnit'] for e in result}
                org_units = {de['ou'] for de in actual_cases['data']}
                assert result_org_units == org_units, (result_org_units, org_units)
            except Exception as err:
                msg = f'Error making evaluation with data for model {model_name}: {err}'
                logger.error(msg)
                errors.append(msg)

        if errors:
            raise Exception('One or more evaluation with data errors, for details see above logs')

    def wait_for_db_id(self, job_id):
        for _ in range(3000):
            job_url = self._chap_url + f"/v1/jobs/{job_id}"
            job_status = self._get(job_url).lower()
            logger.info(job_status)
            if job_status == "failure":
                logs = self._get(job_url + "/logs")
                raise ValueError(f"Failed job: {logs}")
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
    parser.add_argument("model_id", type=str, nargs='?', default='', help="Which model id (name) to test, or leave blank to test all models.")
    parser.add_argument("dataset_path", type=str, nargs='?', default='', help="Path to which dataset will be used for testss.")

    args = parser.parse_args()
    logger.info(args)

    chap_url = f"http://{args.host}:8000"
    suite = IntegrationTest(chap_url, args.model_id, args.dataset_path)
    suite.evaluation_flow()
    # suite.evaluation_with_data_flow() # current create-backtest-with-data.json lacks population data so will only work with naive_model
    suite.prediction_flow()
