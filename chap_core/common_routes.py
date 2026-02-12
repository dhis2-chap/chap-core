"""Common API endpoints shared across all API versions.

These endpoints are mounted at the root level of the application and provide
version-independent functionality such as health checks, status monitoring,
job results retrieval, and compatibility checks.
"""

import logging
from typing import Any, cast

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from packaging.version import Version
from pydantic import BaseModel

from chap_core.api_types import EvaluationResponse
from chap_core.internal_state import Control, InternalState
from chap_core.model_spec import ModelSpec
from chap_core.predictor.feature_spec import Feature
from chap_core.rest_api.celery_tasks import CeleryPool
from chap_core.rest_api.data_models import FullPredictionResponse
from chap_core.rest_api.v1.routers.dependencies import get_settings
from chap_core.worker.interface import SeededJob

logger = logging.getLogger(__name__)


# --- Response models ---


class State(BaseModel):
    """Current state of the job runner."""

    ready: bool
    status: str
    progress: float = 0
    logs: str = ""


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    message: str


class CompatibilityResponse(BaseModel):
    """Response for modelling app compatibility checks."""

    compatible: bool
    description: str


class SystemInfoResponse(BaseModel):
    """System information response."""

    chap_core_version: str
    python_version: str
    os: str


# --- Shared state ---

internal_state = InternalState(Control({}), {})
worker: CeleryPool[Any] = CeleryPool()

# --- Router ---

router = APIRouter()


# -- Health and info endpoints --


@router.get("/health")
async def health(worker_config=Depends(get_settings)) -> HealthResponse:
    """Check that the API is running and healthy."""
    return HealthResponse(status="success", message="healthy")


@router.get("/version")
async def version() -> dict:
    """Return the current chap-core version."""
    from chap_core import __version__ as chap_core_version

    return {"version": chap_core_version}


@router.get("/is-compatible")
async def is_compatible(modelling_app_version: str) -> CompatibilityResponse:
    """Check if a modelling app version is compatible with this API."""
    from chap_core import (
        __minimum_modelling_app_version__ as minimum_modelling_app_version,
    )
    from chap_core import (
        __version__ as chap_core_version,
    )

    if Version(modelling_app_version) < Version(minimum_modelling_app_version):
        return CompatibilityResponse(
            compatible=False,
            description=f"Modelling app version {modelling_app_version} is too old. Minimum version is {minimum_modelling_app_version}",
        )
    return CompatibilityResponse(
        compatible=True,
        description=f"Modelling app version {modelling_app_version} is compatible with the current API version {chap_core_version}",
    )


@router.get("/system-info")
async def system_info() -> SystemInfoResponse:
    """Return system information including versions and OS."""
    import platform

    from chap_core import __version__ as chap_core_version

    return SystemInfoResponse(
        chap_core_version=chap_core_version, python_version=platform.python_version(), os=platform.platform()
    )


# -- Job status and results endpoints --


@router.get("/status")
async def get_status() -> State:
    """Return the current status of the running job."""
    if internal_state.is_ready():
        return State(ready=True, status="idle")

    cur_job = internal_state.current_job
    assert cur_job is not None  # is_ready() returns True if current_job is None
    return State(
        ready=False,
        status=cur_job.status,
        progress=cur_job.progress,
        logs="",
    )


@router.get("/get-results")
async def get_results() -> FullPredictionResponse:
    """Retrieve prediction results from the current job."""
    cur_job = internal_state.current_job
    if cur_job is None:
        raise HTTPException(status_code=400, detail="No job available")
    if cur_job.status == "failed":
        raise HTTPException(status_code=400, detail="Job failed. Check the exception endpoint for more information")
    if not cur_job.is_finished:
        raise HTTPException(status_code=400, detail="No response available")
    return cast(FullPredictionResponse, cur_job.result)


@router.get("/get-evaluation-results")
async def get_evaluation_results() -> EvaluationResponse:
    """Retrieve evaluation results from the current job."""
    cur_job = internal_state.current_job
    if cur_job is None:
        raise HTTPException(status_code=400, detail="No job available")
    if cur_job.status == "failed":
        raise HTTPException(status_code=400, detail="Job failed. Check the exception endpoint for more information")
    if not cur_job.is_finished:
        raise HTTPException(status_code=400, detail="No response available")
    return cast(EvaluationResponse, cur_job.result)


@router.get("/get-exception")
async def get_exception() -> str:
    """Retrieve exception information if the current job failed."""
    cur_job = internal_state.current_job
    if cur_job is None:
        return ""
    return cur_job.exception_info or ""


@router.post("/cancel")
async def cancel() -> dict:
    """Cancel the currently running job."""
    if internal_state.control is not None:
        internal_state.control.cancel()
    return {"status": "success"}


# -- Deprecated endpoints --


@router.get("/list-models", deprecated=True)
async def list_models() -> list[ModelSpec]:
    """List available models. Deprecated: use /v1/crud/model-templates instead."""
    return []


@router.get("/list-features", deprecated=True)
async def list_features() -> list[Feature]:
    """List available features. Deprecated: use model template features instead."""
    return []


# -- Static assets --


@router.get("/favicon.ico", include_in_schema=False)
async def favicon() -> FileResponse:
    return FileResponse("chap_icon.jpeg")


# --- Helpers ---


def seed(data):
    """Seed the internal state with pre-loaded data (used by chap serve)."""
    internal_state.current_job = SeededJob(result=data)
