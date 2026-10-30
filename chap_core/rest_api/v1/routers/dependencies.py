import json
import os
from collections.abc import Iterable
from functools import lru_cache
from typing import TYPE_CHECKING, Any, cast

from fastapi import Depends, HTTPException, Request
from sqlmodel import Session

from chap_core.assessment.target_mask import TARGET_FIELD, check_no_mask_indicator_input
from chap_core.database.database import SessionWrapper, engine
from chap_core.database.dataset_tables import MaskedRange, check_masked_ranges_match_dataset
from chap_core.database.model_templates_and_config_tables import ConfiguredModelDB
from chap_core.database.settings import SettingsManager
from chap_core.rest_api.worker_functions import WorkerConfig

if TYPE_CHECKING:
    from chap_core.spatio_temporal_data.temporal_dataclass import DataSet


# TODO: make dependency injection in celery worker
def get_session():
    with Session(engine) as session:
        yield session


def get_session_wrapper(): ...


def get_settings_manager(session: Session = Depends(get_session)) -> SettingsManager:
    return SettingsManager(session)


@lru_cache
def get_settings():
    return WorkerConfig()


def get_database_url():
    return os.getenv("CHAP_DATABASE_URL")


async def get_job_request(request: Request) -> dict[str, Any]:
    """Capture the submitted JSON body without model defaults or worker credentials."""
    try:
        return cast("dict[str, Any]", json.loads(await request.body()))
    except ValueError:
        # Empty or non-JSON bodies are left to FastAPI's body validation, which returns 422.
        return {}


def get_job_model(session: Session, model_id: int | str) -> ConfiguredModelDB:
    """Resolve the configured model before queuing, so its version can be recorded."""
    try:
        return SessionWrapper(session=session).get_configured_model_by_id_or_name(model_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def model_job_metadata(model: ConfiguredModelDB, parameters: dict, dataset_id: int | None = None) -> dict[str, Any]:
    """Job metadata recorded for a run of a configured model."""
    return {
        "model_id": model.id,
        "model_version": model.model_template.version,
        "dataset_id": dataset_id,
        "parameters": parameters,
    }


def check_masked_ranges_match(masked_ranges: list[MaskedRange] | None, dataset: "DataSet") -> None:
    """Reject masked ranges the data cannot take, with a 422 before anything is queued."""
    if not masked_ranges or not list(dataset.locations()):
        return
    if TARGET_FIELD not in dataset.field_names():
        raise HTTPException(
            status_code=422, detail=f"Masked ranges mask {TARGET_FIELD!r}, but the data has no {TARGET_FIELD!r} column"
        )
    try:
        check_masked_ranges_match_dataset(masked_ranges, dataset.period_range[0])
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


def require_target_column(target_column: str, columns: Iterable[str | None]) -> None:
    """Reject a request whose data has no `target_column`."""
    if target_column not in columns:
        raise HTTPException(status_code=422, detail=f"Dataset has no target column {target_column!r}")


def reject_mask_indicator_input(feature_names: list[str] | list[str | None]) -> None:
    """Reject input data that carries the column masking adds itself."""
    try:
        check_no_mask_indicator_input(feature_names)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
