"""
Integration test for configured models via REST API.

This file tests all configured models that are seeded in the database on startup.
It filters models based on covariate compatibility and period type, then runs
backtest evaluations for each testable model.

This file should not import chap as a python package and should be possible to
run without chap pip installed, only with the necessary requirements (requests).
"""

import requests
import json
import time
import logging
import argparse

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

AVAILABLE_COVARIATES = {"rainfall", "mean_temperature", "population", "disease_cases"}


def load_monthly_backtest_data(filename=None):
    """Load the base backtest data from the example file."""
    if filename is None:
        filename = "example_data/create-backtest-with-data.json"
    return json.load(open(filename))


def add_population_data(data):
    """Add population entries to the provided data."""
    population_entries = [
        entry | {"featureName": "population", "value": 1_000_000}
        for entry in data["providedData"]
        if entry["featureName"] == "rainfall"
    ]
    data["providedData"].extend(population_entries)
    return data


def get_weeks_for_month(year, month):
    """Get the ISO week numbers that fall within a given month.

    Returns list of (year, week) tuples to handle year boundaries correctly.
    """
    import datetime

    weeks = []
    # Get first and last day of the month
    first_day = datetime.date(year, month, 1)
    if month == 12:
        last_day = datetime.date(year + 1, 1, 1) - datetime.timedelta(days=1)
    else:
        last_day = datetime.date(year, month + 1, 1) - datetime.timedelta(days=1)

    # Iterate through the month and collect unique weeks
    current = first_day
    seen_weeks = set()
    while current <= last_day:
        iso_year, iso_week, _ = current.isocalendar()
        if (iso_year, iso_week) not in seen_weeks:
            seen_weeks.add((iso_year, iso_week))
            weeks.append((iso_year, iso_week))
        current += datetime.timedelta(days=1)

    return weeks


def make_weekly_backtest_data(monthly_data):
    """Convert monthly backtest data to weekly format.

    Each monthly data point is expanded to cover all weeks in that month.
    Deduplicates entries where weeks span multiple months.
    """
    data = json.loads(json.dumps(monthly_data))
    new_provided_data = []
    seen = set()

    for entry in data["providedData"]:
        period = entry["period"]
        year = int(period[:4])
        month = int(period[4:6])

        # Get all weeks for this month
        weeks = get_weeks_for_month(year, month)

        # Create an entry for each week
        for iso_year, iso_week in weeks:
            week_period = f"{iso_year}W{iso_week:02d}"
            key = (entry["orgUnit"], entry["featureName"], week_period)

            # Skip if we've already added this combination
            if key in seen:
                continue
            seen.add(key)

            new_entry = entry.copy()
            new_entry["period"] = week_period
            new_provided_data.append(new_entry)

    data["providedData"] = new_provided_data
    return data


def make_backtest_request(model_name, base_data, is_weekly=False):
    """Create a backtest request for a specific model."""
    if is_weekly:
        data = make_weekly_backtest_data(base_data)
    else:
        data = json.loads(json.dumps(base_data))

    data["modelId"] = model_name
    data["name"] = f"integration_test: {model_name}"
    return add_population_data(data)


class ConfiguredModelsIntegrationTest:
    def __init__(self, chap_url):
        self._chap_url = chap_url
        self._base_data = None

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

    def get_configured_models(self):
        """Fetch all configured models from the API."""
        logger.info("Fetching configured models from API")
        models = self._get(self._chap_url + "/v1/crud/models")
        logger.info(f"Found {len(models)} configured models")
        return models

    def filter_testable_models(self, models, monthly_only=False):
        """Filter models based on covariate and period type compatibility."""
        testable = []
        skipped = []

        for model in models:
            name = model["name"]
            period_type = model.get("supportedPeriodType", "any")
            covariates = model.get("covariates", [])
            archived = model.get("archived", False)

            if archived:
                skipped.append((name, "archived"))
                continue

            if monthly_only and period_type == "week":
                skipped.append((name, f"weekly model (--monthly-only flag set)"))
                continue

            covariate_names = {c["name"] for c in covariates}
            missing_covariates = covariate_names - AVAILABLE_COVARIATES
            if missing_covariates:
                skipped.append((name, f"missing covariates: {missing_covariates}"))
                continue

            testable.append(model)

        for name, reason in skipped:
            logger.warning(f"Skipping model '{name}': {reason}")

        logger.info(f"Found {len(testable)} testable models out of {len(models)}")
        return testable

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

    def run_backtest(self, model_name, is_weekly=False):
        """Run a backtest evaluation for a model."""
        logger.info(f"Running backtest for {model_name} (weekly={is_weekly})")
        data = make_backtest_request(model_name, self._base_data, is_weekly=is_weekly)
        response = self._post(f"{self._chap_url}/v1/analytics/create-backtest-with-data/", data)
        job_id = response["id"]
        db_id = self.wait_for_db_id(job_id)
        return db_id

    def verify_backtest_results(self, backtest_id, model_name):
        """Verify the backtest results."""
        logger.info(f"Verifying backtest results for backtest_id={backtest_id}")
        evaluation_result = self._get(self._chap_url + f"/v1/crud/backtests/{backtest_id}/full")
        assert evaluation_result["modelId"] == model_name, (
            f"Expected modelId={model_name}, got {evaluation_result['modelId']}"
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

    def run_test_for_model(self, model):
        """Run the integration test for a single model."""
        model_name = model["name"]
        period_type = model.get("supportedPeriodType", "any")
        is_weekly = period_type == "week"

        logger.info(f"Starting integration test for {model_name} (period_type={period_type})")

        backtest_id = self.run_backtest(model_name, is_weekly=is_weekly)
        self.verify_backtest_results(backtest_id, model_name)

        logger.info(f"SUCCESS: Integration test for {model_name} completed. Backtest ID: {backtest_id}")
        return True

    def run_all_tests(self, model_filter=None, monthly_only=False, data_file=None):
        """Run integration tests for all testable models."""
        logger.info("Starting integration tests for configured models")

        self.ensure_up()
        self._base_data = load_monthly_backtest_data(data_file)

        all_models = self.get_configured_models()

        if model_filter:
            models = [m for m in all_models if m["name"] == model_filter]
            if not models:
                available_names = [m["name"] for m in all_models]
                raise ValueError(f"Model '{model_filter}' not found. Available models: {available_names}")
        else:
            models = self.filter_testable_models(all_models, monthly_only=monthly_only)

        if not models:
            logger.warning("No testable models found")
            return

        results = {"passed": [], "failed": []}

        for model in models:
            model_name = model["name"]
            try:
                self.run_test_for_model(model)
                results["passed"].append(model_name)
            except Exception as e:
                logger.error(f"FAILED: Integration test for {model_name}: {e}")
                results["failed"].append((model_name, str(e)))

        logger.info("=" * 60)
        logger.info("TEST SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Passed: {len(results['passed'])}")
        for name in results["passed"]:
            logger.info(f"  - {name}")

        if results["failed"]:
            logger.info(f"Failed: {len(results['failed'])}")
            for name, error in results["failed"]:
                logger.info(f"  - {name}: {error}")
            raise Exception(f"{len(results['failed'])} test(s) failed")
        else:
            logger.info("SUCCESS: All integration tests completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Integration test for configured models via REST API.")
    parser.add_argument(
        "host",
        type=str,
        nargs="?",
        default="localhost",
        help="Chap REST server host. Defaults to localhost.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Test only a specific model by name.",
    )
    parser.add_argument(
        "--monthly-only",
        action="store_true",
        help="Only test models that support monthly data.",
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default=None,
        help="Path to custom backtest data JSON file.",
    )

    args = parser.parse_args()
    logger.info(args)

    chap_url = f"http://{args.host}:8000"
    test = ConfiguredModelsIntegrationTest(chap_url)
    test.run_all_tests(model_filter=args.model, monthly_only=args.monthly_only, data_file=args.data_file)
