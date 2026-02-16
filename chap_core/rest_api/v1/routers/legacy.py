"""Legacy job-lifecycle endpoints.

These endpoints manage job status, results retrieval, and cancellation.
They are tied to the v1-specific internal_state singleton and response models.
"""

from typing import Any, cast

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from chap_core.api_types import EvaluationResponse
from chap_core.internal_state import Control, InternalState
from chap_core.rest_api.celery_tasks import CeleryPool
from chap_core.rest_api.data_models import FullPredictionResponse
from chap_core.worker.interface import SeededJob


class State(BaseModel):
    """Current state of the job runner."""

    ready: bool
    status: str
    progress: float = 0
    logs: str = ""


internal_state = InternalState(Control({}), {})
worker: CeleryPool[Any] = CeleryPool()

router = APIRouter(tags=["Legacy"])


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
    return cast("FullPredictionResponse", cur_job.result)


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
    return cast("EvaluationResponse", cur_job.result)


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


def seed(data):
    """Seed the internal state with pre-loaded data (used by chap serve)."""
    internal_state.current_job = SeededJob(result=data)
