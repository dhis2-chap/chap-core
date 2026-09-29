"""Unit tests for chap_core.services.alert_service.

Uses an in-memory SQLite engine with foreign-key enforcement enabled so the
referential guards are real, not mocked. These tests cover the service in
isolation; the HTTP-layer wiring is verified by the FastAPI integration tests.
"""

from __future__ import annotations

import pytest
from sqlalchemy import event
from sqlmodel import Session, SQLModel, create_engine

from chap_core.assessment.thresholds.params import SeasonalParams
from chap_core.database.alert_tables import Alert, AlertApproval, AlertLevel
from chap_core.services.alert_policy_service import create_alert_policy
from chap_core.services.alert_service import (
    AlertNotFoundError,
    InvalidAlertError,
    create_alerts,
    get_alert,
    list_alerts,
    set_approval,
)


@pytest.fixture
def engine():
    eng = create_engine("sqlite://")

    @event.listens_for(eng, "connect")
    def _enable_fk(dbapi_connection, _connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    SQLModel.metadata.create_all(eng)
    return eng


@pytest.fixture
def policy_id(engine):
    """A two-tier policy to raise alerts against."""
    with Session(engine) as session:
        policy = create_alert_policy(
            session,
            name="ladder",
            levels=[
                AlertLevel(
                    name="monitor",
                    threshold_params=SeasonalParams(type="seasonal", std_multiplier=1.0),
                    exceedance_threshold=0.3,
                ),
                AlertLevel(
                    name="action",
                    threshold_params=SeasonalParams(type="seasonal", std_multiplier=3.0),
                    exceedance_threshold=0.8,
                ),
            ],
        )
        assert policy.id is not None
        return policy.id


def _alert(policy_id: int, org_unit: str = "A", period: str = "2024-07", level: str = "action") -> Alert:
    return Alert(time_period=period, org_unit=org_unit, alert_policy_id=policy_id, level=level)


def test_created_alerts_start_pending(engine, policy_id):
    """An alert is not released until someone clears it."""
    with Session(engine) as session:
        (alert,) = create_alerts(session, [_alert(policy_id)])
        assert alert.id is not None
        assert alert.approved is AlertApproval.PENDING
        assert alert.approved_by is None
        assert alert.approved_at is None
        assert alert.created is not None


def test_a_level_the_policy_does_not_define_is_rejected(engine, policy_id):
    """A level name only means something against its policy."""
    with Session(engine) as session:
        with pytest.raises(InvalidAlertError, match="no level 'catastrophe'"):
            create_alerts(session, [_alert(policy_id, level="catastrophe")])


def test_an_unknown_policy_is_rejected(engine):
    with Session(engine) as session:
        with pytest.raises(InvalidAlertError, match="AlertPolicy 99999 not found"):
            create_alerts(session, [_alert(99999)])


def test_an_unknown_prediction_is_rejected(engine, policy_id):
    with Session(engine) as session:
        alert = _alert(policy_id)
        alert.prediction_id = 99999
        with pytest.raises(InvalidAlertError, match="Prediction 99999 not found"):
            create_alerts(session, [alert])


def test_a_bad_entry_writes_none_of_the_batch(engine, policy_id):
    """One run raises many alerts, so the batch is validated before anything lands."""
    with Session(engine) as session:
        with pytest.raises(InvalidAlertError):
            create_alerts(session, [_alert(policy_id), _alert(policy_id, level="nonsense")])
        assert list_alerts(session) == []


def test_list_filters_narrow_the_result(engine, policy_id):
    with Session(engine) as session:
        create_alerts(
            session,
            [_alert(policy_id, org_unit="A"), _alert(policy_id, org_unit="B")],
        )
        assert len(list_alerts(session)) == 2
        assert [a.org_unit for a in list_alerts(session, org_unit="B")] == ["B"]
        assert list_alerts(session, alert_policy_id=99999) == []


def test_pending_filter_is_the_review_queue(engine, policy_id):
    with Session(engine) as session:
        first, second = create_alerts(session, [_alert(policy_id, org_unit="A"), _alert(policy_id, org_unit="B")])
        assert first.id is not None
        set_approval(session, [first.id], AlertApproval.APPROVED, "knut")

        queue = list_alerts(session, approved=AlertApproval.PENDING)
        assert [a.org_unit for a in queue] == ["B"]
        released = list_alerts(session, approved=AlertApproval.APPROVED)
        assert [a.org_unit for a in released] == ["A"]


def test_approving_records_who_and_when(engine, policy_id):
    with Session(engine) as session:
        (alert,) = create_alerts(session, [_alert(policy_id)])
        assert alert.id is not None
        (reviewed,) = set_approval(session, [alert.id], AlertApproval.APPROVED, "knut")
        assert reviewed.approved is AlertApproval.APPROVED
        assert reviewed.approved_by == "knut"
        assert reviewed.approved_at is not None


def test_declining_is_distinct_from_pending(engine, policy_id):
    """A declined alert must not drift back into the queue for re-triage."""
    with Session(engine) as session:
        (alert,) = create_alerts(session, [_alert(policy_id)])
        assert alert.id is not None
        set_approval(session, [alert.id], AlertApproval.DECLINED, "knut")

        assert list_alerts(session, approved=AlertApproval.PENDING) == []
        declined = list_alerts(session, approved=AlertApproval.DECLINED)
        assert declined[0].approved_by == "knut"


def test_a_whole_set_is_reviewed_together(engine, policy_id):
    with Session(engine) as session:
        alerts = create_alerts(session, [_alert(policy_id, org_unit=f"OU{i}") for i in range(3)])
        ids = [alert.id for alert in alerts if alert.id is not None]
        reviewed = set_approval(session, ids, AlertApproval.APPROVED, "knut")
        assert {alert.approved for alert in reviewed} == {AlertApproval.APPROVED}


def test_one_unknown_id_leaves_the_whole_set_untouched(engine, policy_id):
    """The reviewer sees a 404 rather than a half-applied decision."""
    with Session(engine) as session:
        (alert,) = create_alerts(session, [_alert(policy_id)])
        assert alert.id is not None
        with pytest.raises(AlertNotFoundError):
            set_approval(session, [alert.id, 99999], AlertApproval.APPROVED, "knut")
        assert get_alert(session, alert.id).approved is AlertApproval.PENDING


def test_reviewing_nothing_is_rejected(engine):
    with Session(engine) as session:
        with pytest.raises(InvalidAlertError, match="No alerts"):
            set_approval(session, [], AlertApproval.APPROVED, "knut")


def test_an_anonymous_review_is_rejected(engine, policy_id):
    with Session(engine) as session:
        (alert,) = create_alerts(session, [_alert(policy_id)])
        assert alert.id is not None
        with pytest.raises(InvalidAlertError, match="approved_by is required"):
            set_approval(session, [alert.id], AlertApproval.APPROVED, "  ")


def test_returning_an_alert_to_the_queue_clears_the_reviewer(engine, policy_id):
    """Sending it back to pending means nobody has decided, so no reviewer is recorded."""
    with Session(engine) as session:
        (alert,) = create_alerts(session, [_alert(policy_id)])
        assert alert.id is not None
        set_approval(session, [alert.id], AlertApproval.APPROVED, "knut")
        (returned,) = set_approval(session, [alert.id], AlertApproval.PENDING, "knut")
        assert returned.approved is AlertApproval.PENDING
        assert returned.approved_by is None
        assert returned.approved_at is None


def test_get_missing_alert_raises_not_found(engine):
    with Session(engine) as session:
        with pytest.raises(AlertNotFoundError):
            get_alert(session, 99999)
