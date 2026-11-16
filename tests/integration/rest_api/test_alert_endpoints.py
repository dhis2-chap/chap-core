import pytest
from sqlmodel import select

from chap_core.database.tables import (
    Backtest,
    Prediction,
)
from tests.integration.rest_api.db_endpoint_helpers import (
    client,
    create_prediction_setup,
    prediction_setup_payload,
)


def _alert_policy_payload(name: str = "Dengue ladder") -> dict:
    return {
        "name": name,
        "levels": [
            {
                "name": "monitor",
                "thresholdParams": {"type": "seasonal", "stdMultiplier": 1.0},
                "exceedanceThreshold": 0.3,
            },
            {
                "name": "action",
                "thresholdParams": {"type": "percentile", "quantile": 0.9},
                "exceedanceThreshold": 0.8,
            },
        ],
    }


def _create_alert_policy(name: str = "Dengue ladder"):
    return client.post("/v1/crud/alert-policies", json=_alert_policy_payload(name))


@pytest.mark.parametrize("codes", [None, ("GREEN", "ORANGE", "RED")])
def test_create_alert_policy_happy_path(clean_engine, dependency_overrides, codes, level_response_actions):
    """A policy round-trips its tiers, each keeping its own strategy's params."""
    payload = _alert_policy_payload("Ladder A")
    payload["levels"][0]["responseAction"] = level_response_actions["monitor"]
    if codes is not None:
        payload["noAlertOutputCode"] = codes[0]
        for level, code in zip(payload["levels"], codes[1:]):
            level["outputCode"] = code
    response = client.post("/v1/crud/alert-policies", json=payload)
    assert response.status_code == 200, response.json()
    policy_id = response.json()["id"]

    body = client.get(f"/v1/crud/alert-policies/{policy_id}").json()
    assert body["name"] == "Ladder A"
    assert [level["name"] for level in body["levels"]] == ["monitor", "action"]
    assert body["levels"][0]["thresholdParams"] == {"type": "seasonal", "stdMultiplier": 1.0}
    assert body["levels"][1]["thresholdParams"]["quantile"] == 0.9
    assert body["levels"][1]["exceedanceThreshold"] == 0.8
    assert body["levels"][0]["responseAction"] == level_response_actions["monitor"]
    assert body["levels"][1]["responseAction"] is None
    assert body["noAlertOutputCode"] == (codes[0] if codes else "0")
    assert [level["outputCode"] for level in body["levels"]] == (list(codes[1:]) if codes else ["1", "2"])
    listed = client.get("/v1/crud/alert-policies").json()
    assert next(policy for policy in listed if policy["id"] == policy_id) == body


@pytest.mark.parametrize(
    "no_alert,monitor,action",
    [
        ("0", "1", "1"),  # duplicate level codes
        ("0", "1", "0"),  # collides with no alert
        ("0", "RED ", "2"),  # padded
    ],
)
def test_create_alert_policy_rejects_invalid_output_codes(
    clean_engine, dependency_overrides, no_alert, monitor, action
):
    payload = _alert_policy_payload()
    payload["noAlertOutputCode"] = no_alert
    payload["levels"][0]["outputCode"] = monitor
    payload["levels"][1]["outputCode"] = action
    response = client.post("/v1/crud/alert-policies", json=payload)
    assert response.status_code == 422, response.text


def test_list_alert_policies_includes_every_created_policy(clean_engine, dependency_overrides):
    # clean_engine is session-scoped, so assert on a subset rather than on an empty list.
    first = _create_alert_policy("Listed one").json()["id"]
    second = _create_alert_policy("Listed two").json()["id"]

    response = client.get("/v1/crud/alert-policies")
    assert response.status_code == 200
    listed = {policy["id"]: policy["name"] for policy in response.json()}
    assert listed[first] == "Listed one"
    assert listed[second] == "Listed two"


def test_get_unknown_alert_policy_returns_404(clean_engine, dependency_overrides):
    assert client.get("/v1/crud/alert-policies/99999").status_code == 404
    assert client.patch("/v1/crud/alert-policies/99999", json={"noAlertOutputCode": "GREEN"}).status_code == 404


def test_create_alert_policy_without_levels_returns_422(clean_engine, dependency_overrides):
    response = client.post("/v1/crud/alert-policies", json={"name": "empty", "levels": []})
    assert response.status_code == 422, response.text


def test_create_alert_policy_with_an_unknown_strategy_returns_422(clean_engine, dependency_overrides):
    """The params discriminator only admits registered strategies."""
    response = client.post(
        "/v1/crud/alert-policies",
        json={
            "name": "bogus",
            "levels": [
                {
                    "name": "alert",
                    "outputCode": "1",
                    "thresholdParams": {"type": "not-a-strategy"},
                    "exceedanceThreshold": 0.5,
                }
            ],
        },
    )
    assert response.status_code == 422, response.text


def test_create_alert_policy_with_out_of_range_exceedance_returns_422(clean_engine, dependency_overrides):
    response = client.post(
        "/v1/crud/alert-policies",
        json={
            "name": "bad cut",
            "levels": [
                {
                    "name": "alert",
                    "outputCode": "1",
                    "thresholdParams": {"type": "seasonal"},
                    "exceedanceThreshold": 1.5,
                }
            ],
        },
    )
    assert response.status_code == 422, response.text


def test_delete_unused_alert_policy(clean_engine, dependency_overrides):
    policy_id = _create_alert_policy("Ladder D").json()["id"]
    assert client.delete(f"/v1/crud/alert-policies/{policy_id}").status_code == 200
    assert client.get(f"/v1/crud/alert-policies/{policy_id}").status_code == 404


def test_prediction_setup_carries_its_alert_policy(override_session, seeded_session):
    """A setup created against a policy reports it back on read."""
    backtest = seeded_session.exec(select(Backtest)).first()
    assert backtest is not None
    policy_id = _create_alert_policy("Ladder E").json()["id"]

    payload = prediction_setup_payload(backtest.id, "Setup with alerts") | {"alertPolicyId": policy_id}
    response = client.post("/v1/crud/prediction-setups", json=payload)
    assert response.status_code == 200, response.json()

    body = client.get(f"/v1/crud/prediction-setups/{response.json()['id']}").json()
    assert body["alertPolicy"]["id"] == policy_id
    assert [level["name"] for level in body["alertPolicy"]["levels"]] == ["monitor", "action"]


def test_prediction_setup_without_a_policy_reports_none(override_session, seeded_session):
    backtest = seeded_session.exec(select(Backtest)).first()
    assert backtest is not None
    setup_id = create_prediction_setup(backtest.id, "No alerts").json()["id"]
    assert client.get(f"/v1/crud/prediction-setups/{setup_id}").json()["alertPolicy"] is None


def test_prediction_setup_with_unknown_alert_policy_returns_404(override_session, seeded_session):
    backtest = seeded_session.exec(select(Backtest)).first()
    assert backtest is not None
    payload = prediction_setup_payload(backtest.id, "ghost policy") | {"alertPolicyId": 99999}
    assert client.post("/v1/crud/prediction-setups", json=payload).status_code == 404


def test_attaching_an_alert_policy_by_patch(override_session, seeded_session):
    backtest = seeded_session.exec(select(Backtest)).first()
    assert backtest is not None
    setup_id = create_prediction_setup(backtest.id, "Later alerts").json()["id"]
    policy_id = _create_alert_policy("Ladder F").json()["id"]

    response = client.patch(f"/v1/crud/prediction-setups/{setup_id}", json={"alertPolicyId": policy_id})
    assert response.status_code == 200, response.json()
    assert response.json()["alertPolicy"]["id"] == policy_id


def _alert_payload(policy_id: int, org_unit: str = "A", period: str = "2024-07", level: str = "action") -> dict:
    return {"timePeriod": period, "orgUnit": org_unit, "alertPolicyId": policy_id, "level": level}


def _create_alerts(policy_id: int, **kwargs):
    return client.post("/v1/crud/alerts", json={"alerts": [_alert_payload(policy_id, **kwargs)]})


def test_create_alert_starts_pending(clean_engine, dependency_overrides):
    """An alert is recorded unreleased; a human has to clear it."""
    policy_id = _create_alert_policy("Alerts A").json()["id"]
    response = _create_alerts(policy_id)
    assert response.status_code == 200, response.text
    (alert_id,) = response.json()["ids"]

    body = client.get(f"/v1/crud/alerts/{alert_id}").json()
    assert body["approved"] == "pending"
    assert body["approvedBy"] is None
    assert body["level"] == "action"
    assert body["alertPolicyId"] == policy_id


def test_create_alert_with_a_level_the_policy_lacks_returns_422(clean_engine, dependency_overrides):
    policy_id = _create_alert_policy("Alerts B").json()["id"]
    response = _create_alerts(policy_id, level="catastrophe")
    assert response.status_code == 422, response.text


def test_get_unknown_alert_returns_404(clean_engine, dependency_overrides):
    assert client.get("/v1/crud/alerts/99999").status_code == 404


def test_approving_a_set_of_alerts(clean_engine, dependency_overrides):
    policy_id = _create_alert_policy("Alerts C").json()["id"]
    created = client.post(
        "/v1/crud/alerts",
        json={"alerts": [_alert_payload(policy_id, org_unit="A"), _alert_payload(policy_id, org_unit="B")]},
    ).json()["ids"]

    response = client.post(
        "/v1/crud/alerts/$approve",
        json={"alertIds": created, "approved": "approved", "approvedBy": "knut"},
    )
    assert response.status_code == 200, response.text
    assert {alert["approved"] for alert in response.json()} == {"approved"}
    assert {alert["approvedBy"] for alert in response.json()} == {"knut"}


def test_approving_with_an_unknown_id_returns_404(clean_engine, dependency_overrides):
    policy_id = _create_alert_policy("Alerts E").json()["id"]
    (alert_id,) = _create_alerts(policy_id).json()["ids"]

    response = client.post(
        "/v1/crud/alerts/$approve",
        json={"alertIds": [alert_id, 99999], "approved": "approved", "approvedBy": "knut"},
    )
    assert response.status_code == 404, response.text


def test_listing_alerts_filters_by_policy(clean_engine, dependency_overrides):
    first_policy = _create_alert_policy("Alerts F").json()["id"]
    second_policy = _create_alert_policy("Alerts G").json()["id"]
    (expected,) = _create_alerts(first_policy).json()["ids"]
    _create_alerts(second_policy)

    listed = client.get("/v1/crud/alerts", params={"alertPolicyId": first_policy}).json()
    assert [alert["id"] for alert in listed] == [expected]


def test_deleting_a_policy_with_alerts_returns_409(clean_engine, dependency_overrides):
    """An alert's level name only means something against its policy."""
    policy_id = _create_alert_policy("Alerts H").json()["id"]
    _create_alerts(policy_id)
    assert client.delete(f"/v1/crud/alert-policies/{policy_id}").status_code == 409


def test_deleting_a_prediction_with_alerts_returns_409(override_session, seeded_session):
    """An alert only means something alongside the forecast that justified it."""
    prediction = seeded_session.exec(select(Prediction)).first()
    assert prediction is not None
    policy_id = _create_alert_policy("Alerts I").json()["id"]
    payload = _alert_payload(policy_id) | {"predictionId": prediction.id}
    assert client.post("/v1/crud/alerts", json={"alerts": [payload]}).status_code == 200

    response = client.delete(f"/v1/crud/predictions/{prediction.id}")
    assert response.status_code == 409, response.text
    assert "alert" in response.json()["detail"].lower()


def test_deleting_an_alert_lets_its_policy_be_deleted(clean_engine, dependency_overrides):
    policy_id = _create_alert_policy("Alerts K").json()["id"]
    (alert_id,) = _create_alerts(policy_id).json()["ids"]

    assert client.delete(f"/v1/crud/alerts/{alert_id}").status_code == 200
    assert client.get(f"/v1/crud/alerts/{alert_id}").status_code == 404
    assert client.delete(f"/v1/crud/alert-policies/{policy_id}").status_code == 200


def test_deleting_an_approved_alert_returns_409(clean_engine, dependency_overrides):
    policy_id = _create_alert_policy("Alerts L").json()["id"]
    (alert_id,) = _create_alerts(policy_id).json()["ids"]
    client.post("/v1/crud/alerts/$approve", json={"alertIds": [alert_id], "approved": "approved", "approvedBy": "knut"})

    assert client.delete(f"/v1/crud/alerts/{alert_id}").status_code == 409


def test_deleting_an_unknown_alert_returns_404(clean_engine, dependency_overrides):
    assert client.delete("/v1/crud/alerts/99999").status_code == 404


def test_deleting_a_prediction_without_alerts_still_works(override_session, seeded_session):
    prediction = seeded_session.exec(select(Prediction)).first()
    assert prediction is not None
    assert client.delete(f"/v1/crud/predictions/{prediction.id}").status_code == 200


def test_patch_alert_policy_preserves_levels_and_existing_alerts(
    clean_engine, dependency_overrides, level_response_actions
):
    policy_id = _create_alert_policy("Alerts J").json()["id"]
    url = f"/v1/crud/alert-policies/{policy_id}"
    before = client.get(url).json()
    (alert_id,) = _create_alerts(policy_id).json()["ids"]
    alert_before = client.get(f"/v1/crud/alerts/{alert_id}").json()

    response = client.patch(
        url,
        json={
            "noAlertOutputCode": "GREEN",
            "levelOutputCodes": {"action": "RED", "monitor": "ORANGE"},
            "levelResponseActions": level_response_actions,
        },
    )
    assert response.status_code == 200, response.text
    expected = before | {"noAlertOutputCode": "GREEN"}
    expected["levels"] = [
        level | {"outputCode": code, "responseAction": level_response_actions[level["name"]]}
        for level, code in zip(before["levels"], ["ORANGE", "RED"])
    ]
    assert response.json() == expected
    assert client.get(url).json() == expected
    alert = client.get(f"/v1/crud/alerts/{alert_id}").json()
    assert alert == alert_before
    assert next(level["outputCode"] for level in expected["levels"] if level["name"] == alert["level"]) == "RED"

    # A partial update preserves both the no-alert code and omitted levels.
    response = client.patch(url, json={"levelOutputCodes": {"monitor": "YELLOW"}})
    expected["levels"][0]["outputCode"] = "YELLOW"
    assert response.status_code == 200, response.text
    assert client.get(url).json() == expected

    # An action-only update can clear one action while preserving the rest of the policy.
    response = client.patch(url, json={"levelResponseActions": {"action": None}})
    expected["levels"][1]["responseAction"] = None
    assert response.status_code == 200, response.text
    assert response.json() == expected
    assert client.get(url).json() == expected


@pytest.mark.parametrize(
    "update",
    [
        {"levelOutputCodes": {"unknown": "RED"}},
        {"levelResponseActions": {"unknown": None}},
        {"levelOutputCodes": {"monitor": "2"}, "levelResponseActions": {"action": "Do not save"}},
        {"levelOutputCodes": {"monitor": "2"}},  # collides with an unchanged level
        {"noAlertOutputCode": "1"},  # collides with an unchanged level
        {"name": "renamed"},  # only codes and response actions are editable
    ],
)
def test_patch_alert_policy_rejects_invalid_changes_without_saving(clean_engine, dependency_overrides, update):
    policy_id = _create_alert_policy().json()["id"]
    url = f"/v1/crud/alert-policies/{policy_id}"
    before = client.get(url).json()
    response = client.patch(url, json=update)
    assert response.status_code == 422, response.text
    assert client.get(url).json() == before


def _test_backtest_id(session) -> int:
    return session.exec(select(Backtest).where(Backtest.name == "test backtest")).one().id


def test_outbreak_metrics_scores_every_level(override_session, seeded_session):
    backtest_id = _test_backtest_id(seeded_session)
    policy_id = _create_alert_policy().json()["id"]

    response = client.get(
        f"/v1/analytics/backtests/{backtest_id}/outbreak-metrics", params={"alertPolicyId": policy_id}
    )

    assert response.status_code == 200, response.json()
    body = response.json()
    assert body["categories"] == ["none", "monitor", "action"]
    (overall,) = body["rows"]
    assert overall["nCells"] > 0
    assert [level["level"] for level in overall["levels"]] == ["monitor", "action"]
    assert "sensitivity" in overall["levels"][0]["metrics"]
    assert "weighted_kappa" in overall["categorical"]


def test_outbreak_metrics_without_computable_thresholds_is_400(override_session, seeded_session):
    """The NaN backtest has three observed cases in all, so no level can compute a threshold before any split."""
    backtest_id = seeded_session.exec(select(Backtest).where(Backtest.name == "test_backtest_with_nans")).one().id
    policy_id = _create_alert_policy().json()["id"]

    response = client.get(
        f"/v1/analytics/backtests/{backtest_id}/outbreak-metrics", params={"alertPolicyId": policy_id}
    )

    assert response.status_code == 400, response.json()


def test_outbreak_metrics_groups_by_the_requested_dimensions(override_session, seeded_session):
    backtest_id = _test_backtest_id(seeded_session)
    policy_id = _create_alert_policy().json()["id"]

    response = client.get(
        f"/v1/analytics/backtests/{backtest_id}/outbreak-metrics",
        params={"alertPolicyId": policy_id, "groupBy": ["location", "horizon_distance"]},
    )

    assert response.status_code == 200, response.json()
    rows = response.json()["rows"]
    groups = [(row["location"], row["horizonDistance"]) for row in rows]
    assert groups == sorted(groups) and len(groups) == 9
    assert all(row["timePeriod"] is None for row in rows)


def test_list_outbreak_metrics_has_unique_ids_of_both_kinds():
    listed = client.get("/v1/analytics/outbreak-metrics").json()
    metrics = {m["id"]: m["kind"] for m in listed}
    assert len(metrics) == len(listed)
    assert metrics["sensitivity"] == "binary"
    assert metrics["weighted_kappa"] == "categorical"


def test_outbreak_metrics_unknown_ids_return_404(override_session, seeded_session):
    backtest_id = _test_backtest_id(seeded_session)
    policy_id = _create_alert_policy().json()["id"]

    unknown_policy = client.get(
        f"/v1/analytics/backtests/{backtest_id}/outbreak-metrics", params={"alertPolicyId": policy_id + 1000}
    )
    unknown_backtest = client.get(
        f"/v1/analytics/backtests/{backtest_id + 1000}/outbreak-metrics", params={"alertPolicyId": policy_id}
    )

    assert unknown_policy.status_code == 404
    assert unknown_backtest.status_code == 404
