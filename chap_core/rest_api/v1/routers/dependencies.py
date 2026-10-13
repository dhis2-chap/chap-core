import json
import os
from functools import lru_cache
from typing import Any, cast

from fastapi import HTTPException, Request
from sqlmodel import Session

from chap_core.database.database import SessionWrapper, engine
from chap_core.database.model_templates_and_config_tables import ConfiguredModelDB
from chap_core.rest_api.worker_functions import WorkerConfig


# TODO: make dependency injection in celery worker
def get_session():
    with Session(engine) as session:
        yield session


def get_session_wrapper(): ...


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
