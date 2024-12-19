from http.client import HTTPException
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from chap_core.api_types import EvaluationResponse
from chap_core.rest_api_src.celery_tasks import CeleryPool
from chap_core.rest_api_src.data_models import FullPredictionResponse

router = APIRouter(prefix="/jobs", tags=["jobs"])
worker = CeleryPool()


@router.get("")
def list_jobs() -> dict[str, Any]:
    """
nn    List all jobs currently in the queue
    """
    return worker.list_jobs()


@router.get("/{job_id}")
def get_job_status(job_id: str) -> str:
    return worker.get_job(job_id).status


@router.get("/{job_id}/prediction_result")
def get_prediction_result(job_id: str) -> FullPredictionResponse:
    return get_result(job_id)


def get_result(job_id):
    job = worker.get_job(job_id)
    print(job_id, job.status, job.result)
    if job.status.lower() == "failed":
        raise HTTPException(status_code=400, detail="Job failed. Check the exception endpoint for more information")

    if not (job and job.is_finished):
        raise HTTPException(status_code=400, detail="No response available")

    return job.result


@router.get("/{job_id}/evaluation_result")
def get_evaluation_result(job_id: str) -> EvaluationResponse:
    return get_result(job_id)


class DataBaseResponse(BaseModel):
    id: int


@router.get("/{job_id}/database_result")
def get_database_result(job_id: str) -> DataBaseResponse:
    result = get_result(job_id)
    return DataBaseResponse(id=result)


'''
Datasets for evaluation are versioned and stored
Datasets for predictions are not necessarily versioned and stored, but sent in each request
Evaluation should be run once, then using continous monitoring after 
Task-id
'''
