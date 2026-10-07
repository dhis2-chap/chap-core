"""Forecast alert selection and persistence, using real stored history and strategies."""

import logging
from unittest.mock import patch

import numpy as np
import pytest
from sqlmodel import Session

from chap_core.assessment.thresholds.params import SeasonalParams
from chap_core.database.alert_tables import Alert, AlertApproval
from chap_core.database.database import SessionWrapper
from chap_core.database.tables import Prediction, PredictionSamplesEntry
from chap_core.datatypes import Samples
from chap_core.rest_api.db_worker_functions import run_prediction
from chap_core.services.alert_service import create_alerts, list_alerts, raise_alerts_for_prediction
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.time_period import PeriodRange


@pytest.mark.parametrize("missing", ["setup", "policy", "levels"])
def test_without_policy_levels_creates_no_alerts(alert_prediction, missing):
    session, prediction = alert_prediction
    if missing == "setup":
        prediction.prediction_setup = None
    elif missing == "policy":
        prediction.prediction_setup.alert_policy = None
    else:
        prediction.prediction_setup.alert_policy.levels = []
    session.commit()
    assert raise_alerts_for_prediction(session, prediction.id) == []
    assert list_alerts(session) == []


def test_most_severe_firing_level_per_cell(alert_prediction):
    session, prediction = alert_prediction
    alerts = raise_alerts_for_prediction(session, prediction.id)
    assert {(a.org_unit, a.time_period, a.level) for a in alerts} == {
        ("A", "2024-01", "monitor"),
        ("A", "2024-02", "action"),
    }
    for alert in alerts:
        assert alert.approved is AlertApproval.PENDING
        assert alert.prediction_id == prediction.id
        assert alert.alert_policy_id == prediction.prediction_setup.alert_policy_id
        assert alert.created is not None
        assert alert.approved_by is None
        assert alert.approved_at is None


def test_severity_uses_policy_order(alert_prediction):
    session, prediction = alert_prediction
    policy = prediction.prediction_setup.alert_policy
    policy.levels = list(reversed(policy.levels))
    session.commit()
    alerts = raise_alerts_for_prediction(session, prediction.id)
    assert [a.level for a in alerts] == ["monitor", "monitor"]


@pytest.mark.parametrize("cutoff, fires", [(0.0, True), (0.5, True), (0.5001, False), (1.0, False)])
def test_exceedance_cutoff_is_inclusive_but_samples_must_be_above_line(alert_prediction, cutoff, fires):
    session, prediction = alert_prediction
    policy = prediction.prediction_setup.alert_policy
    policy.levels = [policy.levels[0].model_copy(update={"exceedance_threshold": cutoff})]
    session.commit()
    alerts = raise_alerts_for_prediction(session, prediction.id)
    # Exactly half of January A's samples are strictly above the median threshold of 15.
    assert any(a.org_unit == "A" and a.time_period == "2024-01" for a in alerts) is fires


@pytest.mark.parametrize("missing", ["all_observations", "location", "season", "one_year", "null_values"])
def test_missing_history_is_skipped_with_warning(alert_prediction, missing, caplog):
    session, prediction = alert_prediction
    policy = prediction.prediction_setup.alert_policy
    if missing == "one_year":
        policy.levels = [policy.levels[0].model_copy(update={"threshold_params": SeasonalParams(type="seasonal")})]
    for observation in list(prediction.dataset.observations):
        if missing == "null_values":
            observation.value = None
        elif (
            missing == "all_observations"
            or (missing == "location" and observation.org_unit == "A")
            or (missing == "season" and observation.period.endswith("-01"))
            or (missing == "one_year" and observation.period.startswith("2022"))
        ):
            session.delete(observation)
    session.commit()
    with caplog.at_level(logging.WARNING):
        alerts = raise_alerts_for_prediction(session, prediction.id)
    if missing == "season":
        assert [(a.org_unit, a.time_period) for a in alerts] == [("A", "2024-02")]
    else:
        assert alerts == []
    assert "Skipping alert level" in caplog.text


def test_strategy_failure_does_not_prevent_other_levels(alert_prediction, caplog):
    session, prediction = alert_prediction
    policy = prediction.prediction_setup.alert_policy
    policy.levels = [
        policy.levels[0],
        policy.levels[1].model_copy(update={"threshold_params": SeasonalParams(type="seasonal")}),
    ]
    session.commit()
    with patch(
        "chap_core.assessment.thresholds.seasonal.SeasonalThresholdStrategy.compute", side_effect=RuntimeError("failed")
    ):
        with caplog.at_level(logging.WARNING):
            alerts = raise_alerts_for_prediction(session, prediction.id)
    assert [a.level for a in alerts] == ["monitor", "monitor"]
    assert "threshold computation failed" in caplog.text


@pytest.mark.parametrize("approval", list(AlertApproval))
def test_running_twice_preserves_existing_alerts(alert_prediction, approval):
    session, prediction = alert_prediction
    alerts = raise_alerts_for_prediction(session, prediction.id)
    ids = [a.id for a in alerts]
    for alert in alerts:
        alert.approved = approval
    session.commit()
    assert raise_alerts_for_prediction(session, prediction.id) == []
    stored = list_alerts(session, prediction_id=prediction.id)
    assert [a.id for a in stored] == ids
    assert all(a.approved is approval for a in stored)


def test_retry_can_add_missing_cells(alert_prediction):
    session, prediction = alert_prediction
    raise_alerts_for_prediction(session, prediction.id)
    prediction.forecasts.append(PredictionSamplesEntry(org_unit="B", period="2024-02", values=[21.0]))
    session.commit()
    alerts = raise_alerts_for_prediction(session, prediction.id)
    assert [(a.org_unit, a.time_period, a.level) for a in alerts] == [("B", "2024-02", "action")]
    assert len(list_alerts(session, prediction_id=prediction.id)) == 3


@pytest.mark.parametrize("fail_alerts", [False, True])
def test_run_prediction_stores_forecast_before_raising_alerts(alert_prediction, fail_alerts, caplog):
    session, original = alert_prediction
    wrapper = SessionWrapper(session=session)
    # Mock model execution only; use the real prediction and alert persistence paths.
    forecast = DataSet({"A": Samples(PeriodRange.from_strings(["2024-01"]), np.array([[21.0, 21.0]]))})

    def fail_commit(session_arg, alerts):
        # A real flush failure leaves the session needing rollback.
        session_arg.add(
            Alert(prediction_id=-1, alert_policy_id=-1, org_unit="A", time_period="2024-01", level="action")
        )
        session_arg.flush()

    with (
        patch.object(wrapper, "get_configured_model_with_code"),
        patch("chap_core.rest_api.db_worker_functions.forecast_ahead", return_value=forecast),
        patch(
            "chap_core.services.alert_service.create_alerts",
            wraps=create_alerts,
            side_effect=fail_commit if fail_alerts else None,
        ),
        caplog.at_level(logging.ERROR),
    ):
        prediction_id = run_prediction(
            original.model_id, str(original.dataset_id), 1, "new", wrapper, original.prediction_setup_id
        )

    with Session(session.get_bind()) as verification:
        stored = verification.get(Prediction, prediction_id)
        assert stored is not None
        assert stored.forecasts[0].values == [21.0, 21.0]
        alerts = list_alerts(verification, prediction_id=prediction_id)
        assert len(alerts) == (0 if fail_alerts else 1)
    # The original session is usable again even after an alert transaction fails.
    assert session.get(Prediction, prediction_id) is not None
    if fail_alerts:
        assert "the stored prediction is kept" in caplog.text
        assert "FOREIGN KEY constraint failed" in caplog.text
