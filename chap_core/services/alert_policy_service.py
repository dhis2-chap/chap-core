"""Service layer for AlertPolicy CRUD operations.

Domain logic that doesn't depend on HTTP concerns. Raises typed exceptions that
the router (or any other caller) maps to its own error format.

There is deliberately no update: a policy's levels cannot be edited in place.
Renaming or dropping a tier would leave every existing alert naming a level the
policy no longer defines, breaking the invariant that `Alert.level` is always
one of its policy's level names. Policies are created and replaced, not edited.
"""

from __future__ import annotations

import datetime
import logging
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from chap_core.database.alert_tables import Alert, AlertLevel, AlertPolicy
from chap_core.database.tables import PredictionSetup

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


class AlertPolicyServiceError(Exception):
    """Base exception for AlertPolicy service errors."""


class AlertPolicyNotFoundError(AlertPolicyServiceError):
    """Raised when the referenced AlertPolicy does not exist."""


class InvalidAlertPolicyError(AlertPolicyServiceError):
    """Raised when input fails validation (empty name, no levels, duplicate level names)."""


class AlertPolicyInUseError(AlertPolicyServiceError):
    """Raised when deleting a policy that a prediction setup or an alert still points at."""


def _validate_levels(levels: Sequence[AlertLevel | dict[str, Any]]) -> list[AlertLevel]:
    """Parse and check a policy's tiers.

    Accepts either models or the plain dicts a partial-update body produces, so
    the service owns the contract wherever the levels came from.

    Raises:
        InvalidAlertPolicyError: no levels, a malformed level, or an empty or
            duplicated tier name.
    """
    if not levels:
        raise InvalidAlertPolicyError("An alert policy needs at least one level")
    try:
        parsed = [level if isinstance(level, AlertLevel) else AlertLevel.model_validate(level) for level in levels]
    except ValidationError as e:
        raise InvalidAlertPolicyError(f"Invalid alert level: {e}") from e
    names = [level.name for level in parsed]
    if not all(name.strip() for name in names):
        raise InvalidAlertPolicyError("Every alert level needs a name")
    padded = [name for name in names if name != name.strip()]
    if padded:
        raise InvalidAlertPolicyError(f"Alert level names cannot start or end with whitespace, got {padded}")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise InvalidAlertPolicyError(f"Alert level names must be unique within a policy, got duplicates: {duplicates}")
    return parsed


def create_alert_policy(session: Session, name: str, levels: Sequence[AlertLevel | dict[str, Any]]) -> AlertPolicy:
    """Create a new AlertPolicy.

    Raises:
        InvalidAlertPolicyError: name or levels fail validation.
    """
    if not name or not name.strip():
        raise InvalidAlertPolicyError("name is required")

    policy = AlertPolicy(name=name, created=datetime.datetime.now(), levels=_validate_levels(levels))
    session.add(policy)
    session.commit()
    session.refresh(policy)
    return policy


def get_alert_policy(session: Session, policy_id: int) -> AlertPolicy:
    """Fetch an AlertPolicy by id. Raises AlertPolicyNotFoundError if missing."""
    policy = session.get(AlertPolicy, policy_id)
    if policy is None:
        raise AlertPolicyNotFoundError(f"AlertPolicy {policy_id} not found")
    return policy


def list_alert_policies(session: Session) -> list[AlertPolicy]:
    return list(session.exec(select(AlertPolicy)).all())


def delete_alert_policy(session: Session, policy_id: int) -> None:
    """Delete an AlertPolicy.

    Raises:
        AlertPolicyNotFoundError: policy does not exist.
        AlertPolicyInUseError: a prediction setup or an alert still points at the
            policy. An alert's level name only means something against its policy,
            so deleting the policy would leave the alert unreadable.
    """
    policy = get_alert_policy(session, policy_id)

    setup_id = session.exec(select(PredictionSetup.id).where(PredictionSetup.alert_policy_id == policy_id)).first()
    if setup_id is not None:
        raise AlertPolicyInUseError(f"AlertPolicy {policy_id} is still used by prediction setup {setup_id}")

    alert_id = session.exec(select(Alert.id).where(Alert.alert_policy_id == policy_id)).first()
    if alert_id is not None:
        raise AlertPolicyInUseError(f"AlertPolicy {policy_id} still has alerts raised against it, e.g. {alert_id}")

    session.delete(policy)
    try:
        session.commit()
    except IntegrityError as e:
        session.rollback()
        raise AlertPolicyInUseError(f"AlertPolicy {policy_id} is still in use") from e
