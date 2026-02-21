"""
Integration test for external models via REST API.

This file tests the flow of registering model templates via the REST API,
creating configured models, and running backtest evaluations.

This file should not import chap as a python package and should be possible to
run without chap pip installed, only with the necessary requirements (requests).
"""

import requests
import json
import time
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MODELS_TO_TEST = [
    ("https://github.com/dhis2-chap/minimalist_example_r", "stable"),
    ("https://github.com/dhis2-chap/minimalist_example_uv", "stable"),
]


def make_backtest_with_data_request(model_name):
    """Load backtest data and set the model name."""
    filename = "example_data/create-backtest-with-data.json"
    data = json.load(open(filename))
    data["modelId"] = model_name
    data["name"] = f"integration_test: {model_name} with data"
    population_entries = [
        entry | {"featureName": "population", "value": 1_000_000}
        for entry in data["providedData"]
        if entry["featureName"] == "rainfall"
    ]
    data["providedData"].extend(population_entries)
    return data


class ModelIntegrationTest:
    def __init__(self, chap_url):
        self._chap_url = chap_url

    def ensure_up(self):
        """Wait for API to be available with retry."""
        response = None
        logger.info("Ensuring %s is up" % self._chap_url)
        errors = []
        for _ in range(40):
            try:
                response = requests.get(self._chap_url + "/v1/health")
                break
            except requests.exceptions.ConnectionError as e:
                logger.error("Failed to connect to %s. Will sleep and try again." % self._chap_url)
                logger.error(e)
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
        except Exception:
            logger.error("Failed to connect to %s" % self._chap_url)
            logger.error("Failed to get %s" % url)
            raise
        assert response.status_code == 200, (response.status_code, response.text)
        return response.json()

    def _post(self, url, json_data):
        try:
            response = requests.post(url, json=json_data)
        except Exception:
            logger.error("Failed to connect to %s" % self._chap_url)
            logger.error("Failed to post to %s" % url)
            raise
        assert response.status_code == 200, (response.status_code, response.text)
        return response.json()

    def register_model_template(self, url, version):
        """Register a model template via REST API."""
        logger.info("Registering model template from %s" % url)
        response = self._post(
            f"{self._chap_url}/v1/crud/model-templates",
            {"url": url, "version": version},
        )
        template_id = response["id"]
        template_name = response["name"]
        logger.info(f"Registered model template with id={template_id}, name={template_name}")
        return template_id, template_name

    def create_configured_model(self, template_id, configured_model_name):
        """Create a configured model from the template."""
        logger.info(f"Creating configured model '{configured_model_name}' from template {template_id}")
        response = self._post(
            f"{self._chap_url}/v1/crud/configured-models",
            {
                "name": configured_model_name,
                "modelTemplateId": template_id,
                "userOptionValues": {},
                "additionalContinuousCovariates": [],
            },
        )
        configured_id = response["id"]
        logger.info(f"Created configured model with id={configured_id}")
        return configured_id

    def wait_for_db_id(self, job_id):
        """Wait for a job to complete and return the database ID."""
        for _ in range(3000):
            job_url = self._chap_url + f"/v1/jobs/{job_id}"
            job_status = self._get(job_url).lower()
            logger.info(f"Job status: {job_status}")
            if job_status == "failure":
                try:
                    logs = self._get(job_url + "/logs")
                except Exception:
                    logs = "Could not get logs"
                raise ValueError(f"Failed job: {logs}")
            if job_status == "success":
                return self._get(job_url + "/database_result/")["id"]
            time.sleep(1)
        raise TimeoutError("Job took too long")

    def run_backtest(self, configured_model_name):
        """Run a backtest evaluation with the configured model."""
        logger.info(f"Running backtest for {configured_model_name}")
        data = make_backtest_with_data_request(configured_model_name)
        response = self._post(f"{self._chap_url}/v1/analytics/create-backtest-with-data/", data)
        job_id = response["id"]
        db_id = self.wait_for_db_id(job_id)
        return db_id

    def verify_backtest_results(self, backtest_id, configured_model_name):
        """Verify the backtest results."""
        logger.info(f"Verifying backtest results for backtest_id={backtest_id}")
        evaluation_result = self._get(self._chap_url + f"/v1/crud/backtests/{backtest_id}/full")
        assert evaluation_result["modelId"] == configured_model_name, (
            f"Expected modelId={configured_model_name}, got {evaluation_result['modelId']}"
        )
        assert evaluation_result["name"].startswith("integration_test"), evaluation_result["name"]
        assert evaluation_result["created"], "Missing created timestamp"

        url_string = self._chap_url + f"/v1/analytics/evaluation-entry?backtestId={backtest_id}&quantiles=0.5"
        evaluation_entries = self._get(url_string)
        assert len(evaluation_entries) > 0, "No evaluation entries found"
        logger.info(f"Found {len(evaluation_entries)} evaluation entries")

        actual_cases = self._get(self._chap_url + f"/v1/analytics/actualCases/{backtest_id}")
        result_org_units = {e["orgUnit"] for e in evaluation_entries}
        org_units = {de["ou"] for de in actual_cases["data"]}
        assert result_org_units == org_units, (result_org_units, org_units)

        return evaluation_entries

    def run_test_for_model(self, model_url, model_version):
        """Run the integration test for a single model."""
        logger.info(f"Starting integration test for {model_url}")

        # Register the model template
        template_id, template_name = self.register_model_template(model_url, model_version)

        # Create a configured model
        configuration_name = f"{template_name}_integration_test"
        self.create_configured_model(template_id, configuration_name)

        # The full model ID used for backtests is "{template_name}:{configuration_name}"
        model_id = f"{template_name}:{configuration_name}"

        # Run backtest
        backtest_id = self.run_backtest(model_id)

        # Verify results
        self.verify_backtest_results(backtest_id, model_id)

        logger.info(f"SUCCESS: Integration test for {template_name} completed. Backtest ID: {backtest_id}")

    def run_all_tests(self):
        """Run integration tests for all models."""
        logger.info("Starting integration tests for all models")

        self.ensure_up()

        for model_url, model_version in MODELS_TO_TEST:
            self.run_test_for_model(model_url, model_version)

        logger.info("SUCCESS: All integration tests completed")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Integration test for external models via REST API.")
    parser.add_argument(
        "host",
        type=str,
        nargs="?",
        default="localhost",
        help="Chap REST server host. Defaults to localhost.",
    )

    args = parser.parse_args()
    logger.info(args)

    chap_url = f"http://{args.host}:8000"
    test = ModelIntegrationTest(chap_url)
    test.run_all_tests()
