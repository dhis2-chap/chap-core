from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from chap_core.api_types import EvaluationResponse
from chap_core.rest_api_src.celery_tasks import CeleryPool, JobDescription
from chap_core.rest_api_src.data_models import FullPredictionResponse

router = APIRouter(prefix="/jobs", tags=["jobs"])
worker = CeleryPool()


@router.get("")
def list_jobs() -> List[JobDescription]:
    """
    List all jobs currently in the queue
    """
    return worker.list_jobs()


def get_job(job_id):
    job = worker.get_job(job_id)
    print(job_id, job.status, job.result)
    if job.status.lower() == "failed":
        raise HTTPException(status_code=400, detail="Job failed. Check the exception endpoint for more information")

    if not (job and job.is_finished):
        raise HTTPException(status_code=400, detail="No response available")

    return job


@router.get("/{job_id}")
def get_job_status(job_id: str) -> str:
    return get_job(job_id).status


@router.get("/{job_id}/logs")
def get_logs(job_id: str) -> str:
    job = get_job(job_id)
    logs = job.get_logs()
    if logs is None:
        raise HTTPException(status_code=400, detail=f"Log file not found for job ID '{job_id}'")
    return logs


@router.get("/{job_id}/prediction_result")
def get_prediction_result(job_id: str) -> FullPredictionResponse:
    return get_job(job_id).result


@router.get("/{job_id}/evaluation_result")
def get_evaluation_result(job_id: str) -> EvaluationResponse:
    return get_job(job_id).result


class DataBaseResponse(BaseModel):
    id: int


@router.get("/{job_id}/database_result")
def get_database_result(job_id: str) -> DataBaseResponse:
    result = get_job(job_id).result
    return DataBaseResponse(id=result)


'''
Datasets for evaluation are versioned and stored
Datasets for predictions are not necessarily versioned and stored, but sent in each request
Evaluation should be run once, then using continous monitoring after 
Task-id
'''
