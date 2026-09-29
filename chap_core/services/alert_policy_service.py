"""Service layer for AlertPolicy CRUD operations.

Domain logic that doesn't depend on HTTP concerns. Raises typed exceptions that
the router (or any other caller) maps to its own error format.
"""

from __future__ import annotations

import datetime
import logging
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from chap_core.database.alert_tables import AlertLevel, AlertPolicy
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
    """Raised when deleting a policy that a prediction setup still points at."""


_MUTABLE_FIELDS = frozenset({"name", "levels"})


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
    names = [level.name.strip() for level in parsed]
    if not all(names):
        raise InvalidAlertPolicyError("Every alert level needs a name")
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


def update_alert_policy(session: Session, policy_id: int, update_data: dict[str, Any]) -> AlertPolicy:
    """Apply a partial update to an AlertPolicy.

    ``update_data`` holds only the fields the caller explicitly set, e.g. from
    pydantic's ``model_dump(exclude_unset=True)``. Levels are replaced whole
    rather than merged: a tier is meaningful only alongside the others.

    Raises:
        AlertPolicyNotFoundError: policy does not exist.
        InvalidAlertPolicyError: an immutable field, or values that fail validation.
    """
    rejected = set(update_data.keys()) - _MUTABLE_FIELDS
    if rejected:
        raise InvalidAlertPolicyError(f"Cannot update immutable fields: {sorted(rejected)}")

    policy = get_alert_policy(session, policy_id)

    if "name" in update_data:
        name = update_data["name"]
        if not name or not name.strip():
            raise InvalidAlertPolicyError("name cannot be null or empty")
        policy.name = name

    if "levels" in update_data:
        levels = update_data["levels"]
        if levels is None:
            raise InvalidAlertPolicyError("levels cannot be null")
        policy.levels = _validate_levels(levels)

    session.add(policy)
    session.commit()
    session.refresh(policy)
    return policy


def delete_alert_policy(session: Session, policy_id: int) -> None:
    """Delete an AlertPolicy.

    Raises:
        AlertPolicyNotFoundError: policy does not exist.
        AlertPolicyInUseError: a prediction setup still points at the policy.
    """
    policy = get_alert_policy(session, policy_id)

    in_use = session.exec(select(PredictionSetup.id).where(PredictionSetup.alert_policy_id == policy_id)).first()
    if in_use is not None:
        raise AlertPolicyInUseError(f"AlertPolicy {policy_id} is still used by prediction setup {in_use}")

    session.delete(policy)
    try:
        session.commit()
    except IntegrityError as e:
        session.rollback()
        raise AlertPolicyInUseError(f"AlertPolicy {policy_id} is still in use") from e
