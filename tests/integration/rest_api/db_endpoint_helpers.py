"""Helpers shared by the REST API database endpoint tests."""

import itertools
import logging
import time

from fastapi.testclient import TestClient
from sqlmodel import select

from chap_core.database.dataset_tables import (
    DataSet,
    MaskedRange,
    ObservationBase,
)
from chap_core.rest_api.app import app
from chap_core.rest_api.data_models import (
    DatasetMakeRequest,
)

logger = logging.getLogger(__name__)


logging.basicConfig(level=logging.INFO)


client = TestClient(app)


def await_result_id(job_id, timeout=30):
    for _ in range(timeout):
        response = client.get(f"/v1/jobs/{job_id}")
        status = response.json()["status"]
        if status == "SUCCESS":
            return client.get(f"/v1/jobs/{job_id}/database_result").json()["id"]
        if status == "FAILURE":
            res = client.get(f"/v1/jobs/{job_id}/logs").json()
            assert False, ("Job failed", response.json(), res)

        logger.info(status)
        time.sleep(1)
    assert False, "Timed out"


def create_make_data_request(example_polygons, fetch_request, proivided_features, periods=None):
    locations = [f.id for f in example_polygons.features]
    periods = periods or [f"{year}-{month:02d}" for year in range(2020, 2024) for month in range(1, 13)]
    combinations = itertools.product(proivided_features, periods, locations)

    request = DatasetMakeRequest(
        name="testing",
        geojson=example_polygons,
        provided_data=[
            ObservationBase(feature_name=e, period=p, value=i, org_unit=l) for i, (e, p, l) in enumerate(combinations)
        ],
        data_to_be_fetched=fetch_request,
    )

    return request


def mask_seeded_dataset(seeded_session, start: str, end: str) -> int:
    """Give the seeded monthly dataset (2020-01 to 2022-12) one masked range and return its id."""
    stored = seeded_session.exec(select(DataSet)).first()
    stored.masked_ranges = [MaskedRange(start=start, end=end)]
    seeded_session.add(stored)
    seeded_session.commit()
    return stored.id


def prediction_setup_payload(backtest_id: int, name: str = "Saved setup") -> dict:
    return {
        "backtestId": backtest_id,
        "name": name,
        "scheduleCronExpression": "0 6 * * 1",
        "scheduleEnabled": True,
        "quantileTargets": [{"quantile": "median", "dataElementId": "DE_MED"}],
    }


def create_prediction_setup(backtest_id: int, name: str = "Saved setup"):
    return client.post("/v1/crud/prediction-setups", json=prediction_setup_payload(backtest_id, name))


def check_rejected_org_units(content, expected_rejections):
    if expected_rejections is not None:
        assert "rejected" in content, content
        rejected_regions = {rejection["orgUnit"] for rejection in content["rejected"]}
        assert rejected_regions == set(expected_rejections), (rejected_regions, expected_rejections)


def drop_one_covariate_period_per_location(request_payload) -> str:
    """Remove the first observation of a covariate for every location, so validation rejects them all."""
    obs = request_payload["provided_data"]
    target_feature = next(e["feature_name"] for e in obs if e["feature_name"] != "disease_cases")
    seen_locations = set()
    pruned = []
    for entry in obs:
        if entry["feature_name"] == target_feature and entry["org_unit"] not in seen_locations:
            seen_locations.add(entry["org_unit"])
            continue
        pruned.append(entry)
    request_payload["provided_data"] = pruned
    return target_feature


def check_all_regions_rejected_dry_run(url, request_payload):
    target_feature = drop_one_covariate_period_per_location(request_payload)
    response = client.post(f"{url}?dryRun=true", json=request_payload)
    assert response.status_code == 200, response.json()
    body = response.json()
    assert body["id"] is None
    assert body["importedCount"] == 0
    assert len(body["rejected"]) > 0
    assert all(r["featureName"] == target_feature for r in body["rejected"])


def get_content(url):
    response = client.get(url)
    content = response.json()
    assert response.status_code == 200, content
    return content
