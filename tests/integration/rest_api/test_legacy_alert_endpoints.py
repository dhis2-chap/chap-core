"""The deprecated alert routes and field names still work after the rename to signals."""

from sqlmodel import select

from chap_core.database.tables import Backtest
from tests.integration.rest_api.db_endpoint_helpers import client, prediction_setup_payload

ALERT_POLICY = {
    "name": "Dengue ladder",
    "noAlertOutputCode": "GREEN",
    "levels": [
        {
            "name": "level_1",
            "outputCode": "RED",
            "thresholdParams": {"type": "percentile", "quantile": 0.9},
            "exceedanceThreshold": 0.8,
        }
    ],
}


def test_alert_routes_use_the_old_names_and_share_data_with_the_signal_routes(clean_engine, dependency_overrides):
    policy_id = client.post("/v1/crud/alert-policies", json=ALERT_POLICY).json()["id"]
    policy = client.get(f"/v1/crud/alert-policies/{policy_id}").json()
    assert policy["noAlertOutputCode"] == "GREEN"
    assert policy["levels"][0]["exceedanceThreshold"] == 0.8
    assert client.get(f"/v1/crud/threshold-policies/{policy_id}").json()["levels"][0]["alarmThreshold"] == 0.8

    alert = {"alertPolicyId": policy_id, "orgUnit": "A", "timePeriod": "2024-07", "level": "level_1"}
    created = client.post("/v1/crud/alerts", json={"alerts": [alert]})
    assert created.status_code == 200, created.text
    (alert_id,) = created.json()["ids"]
    assert client.get(f"/v1/crud/alerts/{alert_id}").json()["approved"] == "pending"

    approved = client.post(
        "/v1/crud/alerts/$approve", json={"alertIds": [alert_id], "approved": "approved", "approvedBy": "knut"}
    )
    assert approved.status_code == 200, approved.text
    assert approved.json()[0]["approvedBy"] == "knut"
    assert approved.json()[0]["approvedAt"] is not None
    assert client.get(f"/v1/crud/signals/{alert_id}").json()["releaseStatus"] == "released"
    listed = client.get("/v1/crud/alerts", params={"alertPolicyId": policy_id, "approved": "approved"}).json()
    assert [(alert["id"], alert["alertPolicyId"]) for alert in listed] == [(alert_id, policy_id)]

    patched = client.patch(f"/v1/crud/alert-policies/{policy_id}", json={"noAlertOutputCode": "OK"})
    assert patched.json()["noAlertOutputCode"] == "OK"
    assert client.delete(f"/v1/crud/alerts/{alert_id}").status_code == 409  # released signals stay on record
    assert client.get("/v1/crud/alerts/99999").status_code == 404


def test_the_alert_routes_are_marked_deprecated():
    paths = client.get("/openapi.json").json()["paths"]
    old = [path for path in paths if path.startswith(("/v1/crud/alerts", "/v1/crud/alert-policies"))]
    assert len(old) == 5
    assert all(operation["deprecated"] for path in old for operation in paths[path].values())


def test_prediction_setup_accepts_and_serves_the_old_policy_field(override_session, seeded_session):
    backtest_id = seeded_session.exec(select(Backtest.id)).first()
    policy_id = client.post("/v1/crud/alert-policies", json=ALERT_POLICY).json()["id"]

    response = client.post(
        "/v1/crud/prediction-setups", json=prediction_setup_payload(backtest_id) | {"alertPolicyId": policy_id}
    )
    assert response.status_code == 200, response.text
    setup = client.get(f"/v1/crud/prediction-setups/{response.json()['id']}").json()
    assert setup["thresholdPolicy"]["id"] == policy_id
    assert setup["alertPolicy"]["levels"][0]["exceedanceThreshold"] == 0.8

    detached = client.patch(f"/v1/crud/prediction-setups/{setup['id']}", json={"alertPolicyId": None})
    assert detached.json()["alertPolicy"] is None

    metrics_url = f"/v1/analytics/backtests/{backtest_id}/outbreak-metrics"
    old, new = (client.get(metrics_url, params={name: policy_id}) for name in ("alertPolicyId", "thresholdPolicyId"))
    assert (old.status_code, old.json()) == (new.status_code, new.json())
    assert client.get(metrics_url).status_code == 422
