"""Service layer for PredictionSetup CRUD operations.

Domain logic that doesn't depend on HTTP concerns. Raises typed
exceptions that the router (or any other caller) maps to its own
error format.
"""

from __future__ import annotations

import datetime
import logging
from typing import Any

from croniter import croniter  # type: ignore[import-untyped]
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import selectinload
from sqlmodel import Session, select

from chap_core.database.dataset_tables import DataSet
from chap_core.database.model_templates_and_config_tables import ConfiguredModelDB
from chap_core.database.tables import Backtest, Prediction, PredictionSetup, QuantileTarget

logger = logging.getLogger(__name__)


class PredictionSetupServiceError(Exception):
    """Base exception for PredictionSetup service errors."""


class BacktestNotFoundError(PredictionSetupServiceError):
    """Raised when the referenced backtest does not exist."""


class PredictionSetupNotFoundError(PredictionSetupServiceError):
    """Raised when the referenced PredictionSetup does not exist."""


class DuplicateSetupError(PredictionSetupServiceError):
    """Raised when a backtest already has a PredictionSetup."""


class InvalidSetupError(PredictionSetupServiceError):
    """Raised when input fails validation (cron format, required fields, immutable fields)."""


_MUTABLE_FIELDS = frozenset({"name", "schedule_cron_expression", "schedule_enabled", "quantile_targets"})


def _normalize_cron_expression(cron_expression: str | None) -> str | None:
    """Trim whitespace; empty string is treated as None."""
    if cron_expression is None:
        return None
    stripped = cron_expression.strip()
    return stripped or None


def _validate_schedule(cron_expression: str | None, enabled: bool) -> str | None:
    """Validate the cron expression against enabled state. Returns normalized expression or None."""
    cron = _normalize_cron_expression(cron_expression)
    if cron is None:
        if enabled:
            raise InvalidSetupError("Enabled schedules require a cron expression")
        return None
    if len(cron.split()) != 5:
        raise InvalidSetupError("Schedule expression must be standard five-field cron")
    try:
        croniter(cron)
    except Exception as e:
        raise InvalidSetupError(f"Invalid schedule expression: {e}") from e
    return cron


def _setup_read_options(include_predictions: bool = False) -> list[Any]:
    """Eager-load options for returning a setup with its parent model template (and optionally predictions)."""
    options: list[Any] = [
        selectinload(PredictionSetup.configured_model).selectinload(ConfiguredModelDB.model_template),  # type: ignore[arg-type]
    ]
    if include_predictions:
        options.append(
            selectinload(PredictionSetup.predictions)  # type: ignore[arg-type]
            .selectinload(Prediction.dataset)  # type: ignore[arg-type]
            .defer(DataSet.geojson)  # type: ignore[arg-type]
        )
        options.append(
            selectinload(PredictionSetup.predictions)  # type: ignore[arg-type]
            .selectinload(Prediction.configured_model)  # type: ignore[arg-type]
            .selectinload(ConfiguredModelDB.model_template)  # type: ignore[arg-type]
        )
    return options


def create_prediction_setup(
    session: Session,
    backtest_id: int,
    name: str,
    schedule_cron_expression: str | None,
    schedule_enabled: bool,
    quantile_targets: list[QuantileTarget],
) -> PredictionSetup:
    """Create a new PredictionSetup, snapshotting fields from the backtest's dataset.

    Raises:
        BacktestNotFoundError: backtest does not exist.
        InvalidSetupError: cron expression or enabled flag is invalid.
        DuplicateSetupError: backtest already has a PredictionSetup.
    """
    if not name:
        raise InvalidSetupError("name is required")
    cron = _validate_schedule(schedule_cron_expression, schedule_enabled)

    backtest = session.exec(
        select(Backtest)
        .where(Backtest.id == backtest_id)
        .options(
            selectinload(Backtest.dataset).defer(DataSet.geojson),  # type: ignore[arg-type]
        )
    ).first()
    if backtest is None:
        raise BacktestNotFoundError(f"Backtest {backtest_id} not found")

    dataset = backtest.dataset
    setup = PredictionSetup(
        name=name,
        created=datetime.datetime.now(),
        backtest_id=backtest_id,
        configured_model_id=backtest.model_db_id,
        start_period=dataset.first_period,
        org_units=dataset.org_units or [],
        covariate_sources=dataset.data_sources or [],
        period_type=dataset.period_type,
        schedule_cron_expression=cron,
        schedule_enabled=schedule_enabled,
        quantile_targets=list(quantile_targets),
    )
    session.add(setup)
    try:
        session.commit()
    except IntegrityError as e:
        session.rollback()
        raise DuplicateSetupError(f"Backtest {backtest_id} already has a PredictionSetup") from e
    session.refresh(setup)
    return setup


def get_prediction_setup(
    session: Session,
    setup_id: int,
    include_predictions: bool = False,
) -> PredictionSetup:
    """Fetch a PredictionSetup by ID. Raises PredictionSetupNotFoundError if missing."""
    setup = session.exec(
        select(PredictionSetup)
        .where(PredictionSetup.id == setup_id)
        .options(*_setup_read_options(include_predictions=include_predictions))
    ).first()
    if setup is None:
        raise PredictionSetupNotFoundError(f"PredictionSetup {setup_id} not found")
    return setup


def list_prediction_setups(session: Session) -> list[PredictionSetup]:
    return list(session.exec(select(PredictionSetup).options(*_setup_read_options())).all())


def update_prediction_setup(
    session: Session,
    setup_id: int,
    update_data: dict[str, Any],
) -> PredictionSetup:
    """Apply a partial update to a PredictionSetup.

    `update_data` is a dict of fields explicitly set by the caller (e.g.,
    Pydantic's ``model_dump(exclude_unset=True)``). Only mutable fields are
    allowed: name, schedule_cron_expression, schedule_enabled, quantile_targets.

    Raises:
        PredictionSetupNotFoundError: setup does not exist.
        InvalidSetupError: validation failure (immutable field, null on required field,
            invalid cron expression, enabled without expression).
    """
    rejected = set(update_data.keys()) - _MUTABLE_FIELDS
    if rejected:
        raise InvalidSetupError(f"Cannot update immutable fields: {sorted(rejected)}")

    setup = session.exec(
        select(PredictionSetup).where(PredictionSetup.id == setup_id).options(*_setup_read_options())
    ).first()
    if setup is None:
        raise PredictionSetupNotFoundError(f"PredictionSetup {setup_id} not found")

    if "name" in update_data:
        if not update_data["name"]:
            raise InvalidSetupError("name cannot be null or empty")
        setup.name = update_data["name"]

    if "schedule_cron_expression" in update_data or "schedule_enabled" in update_data:
        new_cron = update_data.get("schedule_cron_expression", setup.schedule_cron_expression)
        new_enabled = update_data.get("schedule_enabled", setup.schedule_enabled)
        if new_enabled is None:
            raise InvalidSetupError("schedule_enabled cannot be null")
        validated_cron = _validate_schedule(new_cron, new_enabled)
        setup.schedule_cron_expression = validated_cron
        setup.schedule_enabled = new_enabled

    if "quantile_targets" in update_data:
        targets = update_data["quantile_targets"]
        setup.quantile_targets = list(targets) if targets is not None else []

    session.add(setup)
    session.commit()
    session.refresh(setup)
    return setup


def delete_prediction_setup(session: Session, setup_id: int) -> None:
    """Hard-delete a PredictionSetup.

    The DB-level ON DELETE SET NULL on prediction.prediction_setup_id ensures
    predictions made via this setup are retained with a NULL link.

    Raises:
        PredictionSetupNotFoundError: setup does not exist.
    """
    setup = session.get(PredictionSetup, setup_id)
    if setup is None:
        raise PredictionSetupNotFoundError(f"PredictionSetup {setup_id} not found")
    session.delete(setup)
    session.commit()
