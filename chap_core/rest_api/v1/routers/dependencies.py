import json
import os
from functools import lru_cache
from typing import Any, cast

from fastapi import Request
from sqlmodel import Session

from chap_core.database.database import engine
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
