"""Service layer for Alert CRUD and the release gate.

Domain logic that doesn't depend on HTTP concerns. Raises typed exceptions that
the router (or any other caller) maps to its own error format.
"""

from __future__ import annotations

import datetime
import logging
from typing import TYPE_CHECKING

from sqlmodel import Session, select

from chap_core.database.alert_tables import Alert, AlertApproval, AlertPolicy
from chap_core.database.tables import Prediction

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


class AlertServiceError(Exception):
    """Base exception for Alert service errors."""


class AlertNotFoundError(AlertServiceError):
    """Raised when a referenced Alert does not exist."""


class InvalidAlertError(AlertServiceError):
    """Raised when input fails validation (unknown policy, unknown level, unknown prediction)."""


def _validate_level(session: Session, alert_policy_id: int, level: str) -> None:
    """Check the policy exists and actually defines the named level.

    A level name only means something against its policy, so an alert naming a
    tier the policy does not have would be unreadable later.
    """
    policy = session.get(AlertPolicy, alert_policy_id)
    if policy is None:
        raise InvalidAlertError(f"AlertPolicy {alert_policy_id} not found")
    known = [policy_level.name for policy_level in policy.levels]
    if level not in known:
        raise InvalidAlertError(f"AlertPolicy {alert_policy_id} has no level {level!r}; it defines {sorted(known)}")


def create_alerts(session: Session, alerts: Sequence[Alert]) -> list[Alert]:
    """Record a batch of alerts.

    Alerts arrive in batches -- one run raises many at once -- so the whole batch
    is validated before anything is written and fails together.

    Raises:
        InvalidAlertError: unknown policy, a level the policy does not define, or
            an unknown prediction.
    """
    if not alerts:
        return []

    for alert in alerts:
        _validate_level(session, alert.alert_policy_id, alert.level)
        if alert.prediction_id is not None and session.get(Prediction, alert.prediction_id) is None:
            raise InvalidAlertError(f"Prediction {alert.prediction_id} not found")

    now = datetime.datetime.now()
    for alert in alerts:
        alert.created = now
        session.add(alert)
    session.commit()
    for alert in alerts:
        session.refresh(alert)
    return list(alerts)


def get_alert(session: Session, alert_id: int) -> Alert:
    """Fetch an Alert by id. Raises AlertNotFoundError if missing."""
    alert = session.get(Alert, alert_id)
    if alert is None:
        raise AlertNotFoundError(f"Alert {alert_id} not found")
    return alert


def list_alerts(
    session: Session,
    alert_policy_id: int | None = None,
    prediction_id: int | None = None,
    org_unit: str | None = None,
    approved: AlertApproval | None = None,
) -> list[Alert]:
    """List alerts, narrowed by whichever filters are given.

    Filtering on ``approved=pending`` is the review queue.
    """
    statement = select(Alert)
    if alert_policy_id is not None:
        statement = statement.where(Alert.alert_policy_id == alert_policy_id)
    if prediction_id is not None:
        statement = statement.where(Alert.prediction_id == prediction_id)
    if org_unit is not None:
        statement = statement.where(Alert.org_unit == org_unit)
    if approved is not None:
        statement = statement.where(Alert.approved == approved)
    return list(session.exec(statement).all())


def set_approval(
    session: Session,
    alert_ids: Sequence[int],
    approved: AlertApproval,
    approved_by: str,
) -> list[Alert]:
    """Move a set of alerts through the release gate together.

    Reviewers work through a queue and decide on several alerts at once, so the
    whole set is applied atomically: if any id is unknown, nothing is written.

    Raises:
        AlertNotFoundError: any id does not exist.
        InvalidAlertError: no ids, or an empty reviewer.
    """
    if not alert_ids:
        raise InvalidAlertError("No alerts to review")
    if not approved_by or not approved_by.strip():
        raise InvalidAlertError("approved_by is required")

    alerts = [get_alert(session, alert_id) for alert_id in alert_ids]

    reviewed_at = datetime.datetime.now() if approved is not AlertApproval.PENDING else None
    reviewer = approved_by if approved is not AlertApproval.PENDING else None
    for alert in alerts:
        alert.approved = approved
        alert.approved_by = reviewer
        alert.approved_at = reviewed_at
        session.add(alert)
    session.commit()
    for alert in alerts:
        session.refresh(alert)
    return alerts
