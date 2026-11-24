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


def _threshold_policy_payload(name: str = "Dengue ladder") -> dict:
    return {
        "name": name,
        "levels": [
            {
                "name": "level_1",
                "thresholdParams": {"type": "seasonal", "stdMultiplier": 1.0},
                "alarmThreshold": 0.3,
            },
            {
                "name": "level_2",
                "thresholdParams": {"type": "percentile", "quantile": 0.9},
                "alarmThreshold": 0.8,
            },
        ],
    }


def _create_threshold_policy(name: str = "Dengue ladder"):
    return client.post("/v1/crud/threshold-policies", json=_threshold_policy_payload(name))


@pytest.mark.parametrize("codes", [None, ("GREEN", "ORANGE", "RED")])
def test_create_threshold_policy_happy_path(clean_engine, dependency_overrides, codes, level_response_actions):
    """A policy round-trips its tiers, each keeping its own strategy's params."""
    payload = _threshold_policy_payload("Ladder A")
    payload["levels"][0]["responseAction"] = level_response_actions["level_1"]
    if codes is not None:
        payload["noSignalOutputCode"] = codes[0]
        for level, code in zip(payload["levels"], codes[1:]):
            level["outputCode"] = code
    response = client.post("/v1/crud/threshold-policies", json=payload)
    assert response.status_code == 200, response.json()
    policy_id = response.json()["id"]

    body = client.get(f"/v1/crud/threshold-policies/{policy_id}").json()
    assert body["name"] == "Ladder A"
    assert [level["name"] for level in body["levels"]] == ["level_1", "level_2"]
    assert body["levels"][0]["thresholdParams"] == {"type": "seasonal", "stdMultiplier": 1.0}
    assert body["levels"][1]["thresholdParams"]["quantile"] == 0.9
    assert body["levels"][1]["alarmThreshold"] == 0.8
    assert body["levels"][0]["responseAction"] == level_response_actions["level_1"]
    assert body["levels"][1]["responseAction"] is None
    assert body["noSignalOutputCode"] == (codes[0] if codes else "0")
    assert [level["outputCode"] for level in body["levels"]] == (list(codes[1:]) if codes else ["1", "2"])
    listed = client.get("/v1/crud/threshold-policies").json()
    assert next(policy for policy in listed if policy["id"] == policy_id) == body


@pytest.mark.parametrize(
    "no_signal,monitor,action",
    [
        ("0", "1", "1"),  # duplicate level codes
        ("0", "1", "0"),  # collides with no signal
        ("0", "RED ", "2"),  # padded
    ],
)
def test_create_threshold_policy_rejects_invalid_output_codes(
    clean_engine, dependency_overrides, no_signal, monitor, action
):
    payload = _threshold_policy_payload()
    payload["noSignalOutputCode"] = no_signal
    payload["levels"][0]["outputCode"] = monitor
    payload["levels"][1]["outputCode"] = action
    response = client.post("/v1/crud/threshold-policies", json=payload)
    assert response.status_code == 422, response.text


def test_list_threshold_policies_includes_every_created_policy(clean_engine, dependency_overrides):
    # clean_engine is session-scoped, so assert on a subset rather than on an empty list.
    first = _create_threshold_policy("Listed one").json()["id"]
    second = _create_threshold_policy("Listed two").json()["id"]

    response = client.get("/v1/crud/threshold-policies")
    assert response.status_code == 200
    listed = {policy["id"]: policy["name"] for policy in response.json()}
    assert listed[first] == "Listed one"
    assert listed[second] == "Listed two"


def test_get_unknown_threshold_policy_returns_404(clean_engine, dependency_overrides):
    assert client.get("/v1/crud/threshold-policies/99999").status_code == 404
    assert client.patch("/v1/crud/threshold-policies/99999", json={"noSignalOutputCode": "GREEN"}).status_code == 404


def test_create_threshold_policy_without_levels_returns_422(clean_engine, dependency_overrides):
    response = client.post("/v1/crud/threshold-policies", json={"name": "empty", "levels": []})
    assert response.status_code == 422, response.text


def test_create_threshold_policy_with_an_unknown_strategy_returns_422(clean_engine, dependency_overrides):
    """The params discriminator only admits registered strategies."""
    response = client.post(
        "/v1/crud/threshold-policies",
        json={
            "name": "bogus",
            "levels": [
                {
                    "name": "signal",
                    "outputCode": "1",
                    "thresholdParams": {"type": "not-a-strategy"},
                    "alarmThreshold": 0.5,
                }
            ],
        },
    )
    assert response.status_code == 422, response.text


def test_create_threshold_policy_with_out_of_range_exceedance_returns_422(clean_engine, dependency_overrides):
    response = client.post(
        "/v1/crud/threshold-policies",
        json={
            "name": "bad cut",
            "levels": [
                {
                    "name": "signal",
                    "outputCode": "1",
                    "thresholdParams": {"type": "seasonal"},
                    "alarmThreshold": 1.5,
                }
            ],
        },
    )
    assert response.status_code == 422, response.text


def test_delete_unused_threshold_policy(clean_engine, dependency_overrides):
    policy_id = _create_threshold_policy("Ladder D").json()["id"]
    assert client.delete(f"/v1/crud/threshold-policies/{policy_id}").status_code == 200
    assert client.get(f"/v1/crud/threshold-policies/{policy_id}").status_code == 404


def test_prediction_setup_carries_its_threshold_policy(override_session, seeded_session):
    """A setup created against a policy reports it back on read."""
    backtest = seeded_session.exec(select(Backtest)).first()
    assert backtest is not None
    policy_id = _create_threshold_policy("Ladder E").json()["id"]

    payload = prediction_setup_payload(backtest.id, "Setup with signals") | {"thresholdPolicyId": policy_id}
    response = client.post("/v1/crud/prediction-setups", json=payload)
    assert response.status_code == 200, response.json()

    body = client.get(f"/v1/crud/prediction-setups/{response.json()['id']}").json()
    assert body["thresholdPolicy"]["id"] == policy_id
    assert [level["name"] for level in body["thresholdPolicy"]["levels"]] == ["level_1", "level_2"]


def test_prediction_setup_without_a_policy_reports_none(override_session, seeded_session):
    backtest = seeded_session.exec(select(Backtest)).first()
    assert backtest is not None
    setup_id = create_prediction_setup(backtest.id, "No signals").json()["id"]
    assert client.get(f"/v1/crud/prediction-setups/{setup_id}").json()["thresholdPolicy"] is None


def test_prediction_setup_with_unknown_threshold_policy_returns_404(override_session, seeded_session):
    backtest = seeded_session.exec(select(Backtest)).first()
    assert backtest is not None
    payload = prediction_setup_payload(backtest.id, "ghost policy") | {"thresholdPolicyId": 99999}
    assert client.post("/v1/crud/prediction-setups", json=payload).status_code == 404


def test_attaching_a_threshold_policy_by_patch(override_session, seeded_session):
    backtest = seeded_session.exec(select(Backtest)).first()
    assert backtest is not None
    setup_id = create_prediction_setup(backtest.id, "Later signals").json()["id"]
    policy_id = _create_threshold_policy("Ladder F").json()["id"]

    response = client.patch(f"/v1/crud/prediction-setups/{setup_id}", json={"thresholdPolicyId": policy_id})
    assert response.status_code == 200, response.json()
    assert response.json()["thresholdPolicy"]["id"] == policy_id


def _signal_payload(policy_id: int, org_unit: str = "A", period: str = "2024-07", level: str = "level_2") -> dict:
    return {"timePeriod": period, "orgUnit": org_unit, "thresholdPolicyId": policy_id, "level": level}


def _create_signals(policy_id: int, **kwargs):
    return client.post("/v1/crud/signals", json={"signals": [_signal_payload(policy_id, **kwargs)]})


def test_create_signal_starts_pending(clean_engine, dependency_overrides):
    """A signal is recorded unreleased; a human has to clear it."""
    policy_id = _create_threshold_policy("Signals A").json()["id"]
    response = _create_signals(policy_id)
    assert response.status_code == 200, response.text
    (signal_id,) = response.json()["ids"]

    body = client.get(f"/v1/crud/signals/{signal_id}").json()
    assert body["releaseStatus"] == "pending"
    assert body["reviewedBy"] is None
    assert body["level"] == "level_2"
    assert body["thresholdPolicyId"] == policy_id


def test_create_signal_with_a_level_the_policy_lacks_returns_422(clean_engine, dependency_overrides):
    policy_id = _create_threshold_policy("Signals B").json()["id"]
    response = _create_signals(policy_id, level="catastrophe")
    assert response.status_code == 422, response.text


def test_get_unknown_signal_returns_404(clean_engine, dependency_overrides):
    assert client.get("/v1/crud/signals/99999").status_code == 404


@pytest.mark.parametrize("release_status", ["released", "withheld", "pending"])
def test_setting_signal_release_status(clean_engine, dependency_overrides, release_status):
    policy_id = _create_threshold_policy("Signals C").json()["id"]
    created = client.post(
        "/v1/crud/signals",
        json={"signals": [_signal_payload(policy_id, org_unit="A"), _signal_payload(policy_id, org_unit="B")]},
    ).json()["ids"]

    response = client.post(
        "/v1/crud/signals/$set-release-status",
        json={"signalIds": created, "releaseStatus": release_status, "reviewedBy": "knut"},
    )
    assert response.status_code == 200, response.text
    assert {signal["releaseStatus"] for signal in response.json()} == {release_status}
    assert {signal["reviewedBy"] for signal in response.json()} == {None if release_status == "pending" else "knut"}
    assert all((signal["reviewedAt"] is None) == (release_status == "pending") for signal in response.json())
    listed = client.get("/v1/crud/signals", params={"thresholdPolicyId": policy_id, "releaseStatus": release_status})
    assert listed.status_code == 200
    assert {signal["id"] for signal in listed.json()} == set(created)
    queue = client.get("/v1/crud/signals", params={"thresholdPolicyId": policy_id, "releaseStatus": "pending"}).json()
    assert {signal["id"] for signal in queue} == (set(created) if release_status == "pending" else set())


def test_releasing_with_an_unknown_id_returns_404(clean_engine, dependency_overrides):
    policy_id = _create_threshold_policy("Signals E").json()["id"]
    (signal_id,) = _create_signals(policy_id).json()["ids"]

    response = client.post(
        "/v1/crud/signals/$set-release-status",
        json={"signalIds": [signal_id, 99999], "releaseStatus": "released", "reviewedBy": "knut"},
    )
    assert response.status_code == 404, response.text


def test_listing_signals_filters_by_policy(clean_engine, dependency_overrides):
    first_policy = _create_threshold_policy("Signals F").json()["id"]
    second_policy = _create_threshold_policy("Signals G").json()["id"]
    (expected,) = _create_signals(first_policy).json()["ids"]
    _create_signals(second_policy)

    listed = client.get("/v1/crud/signals", params={"thresholdPolicyId": first_policy}).json()
    assert [signal["id"] for signal in listed] == [expected]


def test_deleting_a_policy_with_signals_returns_409(clean_engine, dependency_overrides):
    """A signal's level name only means something against its policy."""
    policy_id = _create_threshold_policy("Signals H").json()["id"]
    _create_signals(policy_id)
    assert client.delete(f"/v1/crud/threshold-policies/{policy_id}").status_code == 409


def test_deleting_a_prediction_with_signals_returns_409(override_session, seeded_session):
    """A signal only means something alongside the forecast that justified it."""
    prediction = seeded_session.exec(select(Prediction)).first()
    assert prediction is not None
    policy_id = _create_threshold_policy("Signals I").json()["id"]
    payload = _signal_payload(policy_id) | {"predictionId": prediction.id}
    assert client.post("/v1/crud/signals", json={"signals": [payload]}).status_code == 200

    response = client.delete(f"/v1/crud/predictions/{prediction.id}")
    assert response.status_code == 409, response.text
    assert "signal" in response.json()["detail"].lower()


def test_deleting_a_signal_lets_its_policy_be_deleted(clean_engine, dependency_overrides):
    policy_id = _create_threshold_policy("Signals K").json()["id"]
    (signal_id,) = _create_signals(policy_id).json()["ids"]

    assert client.delete(f"/v1/crud/signals/{signal_id}").status_code == 200
    assert client.get(f"/v1/crud/signals/{signal_id}").status_code == 404
    assert client.delete(f"/v1/crud/threshold-policies/{policy_id}").status_code == 200


def test_deleting_a_released_signal_returns_409(clean_engine, dependency_overrides):
    policy_id = _create_threshold_policy("Signals L").json()["id"]
    (signal_id,) = _create_signals(policy_id).json()["ids"]
    client.post(
        "/v1/crud/signals/$set-release-status",
        json={"signalIds": [signal_id], "releaseStatus": "released", "reviewedBy": "knut"},
    )

    assert client.delete(f"/v1/crud/signals/{signal_id}").status_code == 409


def test_deleting_an_unknown_signal_returns_404(clean_engine, dependency_overrides):
    assert client.delete("/v1/crud/signals/99999").status_code == 404


def test_deleting_a_prediction_without_signals_still_works(override_session, seeded_session):
    prediction = seeded_session.exec(select(Prediction)).first()
    assert prediction is not None
    assert client.delete(f"/v1/crud/predictions/{prediction.id}").status_code == 200


def test_patch_threshold_policy_preserves_levels_and_existing_signals(
    clean_engine, dependency_overrides, level_response_actions
):
    policy_id = _create_threshold_policy("Signals J").json()["id"]
    url = f"/v1/crud/threshold-policies/{policy_id}"
    before = client.get(url).json()
    (signal_id,) = _create_signals(policy_id).json()["ids"]
    signal_before = client.get(f"/v1/crud/signals/{signal_id}").json()

    response = client.patch(
        url,
        json={
            "noSignalOutputCode": "GREEN",
            "levelOutputCodes": {"level_2": "RED", "level_1": "ORANGE"},
            "levelResponseActions": level_response_actions,
        },
    )
    assert response.status_code == 200, response.text
    expected = before | {"noSignalOutputCode": "GREEN"}
    expected["levels"] = [
        level | {"outputCode": code, "responseAction": level_response_actions[level["name"]]}
        for level, code in zip(before["levels"], ["ORANGE", "RED"])
    ]
    assert response.json() == expected
    assert client.get(url).json() == expected
    signal = client.get(f"/v1/crud/signals/{signal_id}").json()
    assert signal == signal_before
    assert next(level["outputCode"] for level in expected["levels"] if level["name"] == signal["level"]) == "RED"

    # A partial update preserves both the no-signal code and omitted levels.
    response = client.patch(url, json={"levelOutputCodes": {"level_1": "YELLOW"}})
    expected["levels"][0]["outputCode"] = "YELLOW"
    assert response.status_code == 200, response.text
    assert client.get(url).json() == expected

    # An action-only update can clear one action while preserving the rest of the policy.
    response = client.patch(url, json={"levelResponseActions": {"level_2": None}})
    expected["levels"][1]["responseAction"] = None
    assert response.status_code == 200, response.text
    assert response.json() == expected
    assert client.get(url).json() == expected


@pytest.mark.parametrize(
    "update",
    [
        {"levelOutputCodes": {"unknown": "RED"}},
        {"levelResponseActions": {"unknown": None}},
        {"levelOutputCodes": {"level_1": "2"}, "levelResponseActions": {"level_2": "Do not save"}},
        {"levelOutputCodes": {"level_1": "2"}},  # collides with an unchanged level
        {"noSignalOutputCode": "1"},  # collides with an unchanged level
        {"name": "renamed"},  # only codes and response actions are editable
    ],
)
def test_patch_threshold_policy_rejects_invalid_changes_without_saving(clean_engine, dependency_overrides, update):
    policy_id = _create_threshold_policy().json()["id"]
    url = f"/v1/crud/threshold-policies/{policy_id}"
    before = client.get(url).json()
    response = client.patch(url, json=update)
    assert response.status_code == 422, response.text
    assert client.get(url).json() == before


def _test_backtest_id(session) -> int:
    return session.exec(select(Backtest).where(Backtest.name == "test backtest")).one().id


def test_outbreak_metrics_scores_every_level(override_session, seeded_session):
    backtest_id = _test_backtest_id(seeded_session)
    policy_id = _create_threshold_policy().json()["id"]

    response = client.get(
        f"/v1/analytics/backtests/{backtest_id}/outbreak-metrics", params={"thresholdPolicyId": policy_id}
    )

    assert response.status_code == 200, response.json()
    body = response.json()
    assert body["categories"] == ["none", "level_1", "level_2"]
    (overall,) = body["rows"]
    assert overall["nCells"] > 0
    assert [level["level"] for level in overall["levels"]] == ["level_1", "level_2"]
    assert "sensitivity" in overall["levels"][0]["metrics"]
    assert "weighted_kappa" in overall["categorical"]


def test_outbreak_metrics_without_computable_thresholds_is_400(override_session, seeded_session):
    """The NaN backtest has three observed cases in all, so no level can compute a threshold before any split."""
    backtest_id = seeded_session.exec(select(Backtest).where(Backtest.name == "test_backtest_with_nans")).one().id
    policy_id = _create_threshold_policy().json()["id"]

    response = client.get(
        f"/v1/analytics/backtests/{backtest_id}/outbreak-metrics", params={"thresholdPolicyId": policy_id}
    )

    assert response.status_code == 400, response.json()


def test_outbreak_metrics_groups_by_the_requested_dimensions(override_session, seeded_session):
    backtest_id = _test_backtest_id(seeded_session)
    policy_id = _create_threshold_policy().json()["id"]

    response = client.get(
        f"/v1/analytics/backtests/{backtest_id}/outbreak-metrics",
        params={"thresholdPolicyId": policy_id, "groupBy": ["location", "horizon_distance"]},
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
    policy_id = _create_threshold_policy().json()["id"]

    unknown_policy = client.get(
        f"/v1/analytics/backtests/{backtest_id}/outbreak-metrics", params={"thresholdPolicyId": policy_id + 1000}
    )
    unknown_backtest = client.get(
        f"/v1/analytics/backtests/{backtest_id + 1000}/outbreak-metrics", params={"thresholdPolicyId": policy_id}
    )

    assert unknown_policy.status_code == 404
    assert unknown_backtest.status_code == 404
