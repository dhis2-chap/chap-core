import logging
from typing import Any, cast

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlmodel import Session

from chap_core.database.tables import Backtest, BacktestRead, Prediction, PredictionInfo
from chap_core.log_config import initialize_logging
from chap_core.rest_api.celery_tasks import CeleryPool, JobDescription, get_job_meta
from chap_core.rest_api.celery_tasks import r as redis
from chap_core.rest_api.v1.routers.dependencies import get_session

initialize_logging()
logger = logging.getLogger(__name__)
logger.info("Logging initialized")


router = APIRouter(prefix="/jobs", tags=["Jobs"])
worker: CeleryPool[Any] = CeleryPool()


@router.get(
    "",
    summary="Track background work",
)
def list_jobs(
    ids: list[str] = Query(None),
    status: list[str] = Query(None),
    job_type: str = Query(None, alias="type"),
    prediction_setup_id: int | None = Query(None, alias="predictionSetupId"),
) -> list[JobDescription]:
    """See every backtest, prediction, or dataset-import job currently being tracked, so a UI can render a progress dashboard or a script can wait for a batch to finish.

    Optionally narrow the list by ``ids``, ``type`` (e.g. ``PREDICTION``, ``EVALUATION``),
    ``predictionSetupId`` (every job a given setup has launched), and/or ``status``
    (case-insensitive). Filters compose. An empty list just means nothing matched — no
    404.
    """
    jobs_to_return = worker.list_jobs()

    if ids:
        id_filter_set = set(ids)
        jobs_to_return = [job for job in jobs_to_return if job.id in id_filter_set]

    if job_type:
        type_upper = job_type.upper()
        jobs_to_return = [job for job in jobs_to_return if job.type and job.type.upper() == type_upper]

    if prediction_setup_id is not None:
        jobs_to_return = [job for job in jobs_to_return if job.prediction_setup_id == prediction_setup_id]

    if status:
        status_filter_set = {s.upper() for s in status}
        jobs_to_return = [job for job in jobs_to_return if job.status and job.status.upper() in status_filter_set]

    return cast("list[JobDescription]", jobs_to_return)


def _ensure_job_exists(job_id: str) -> None:
    """Celery's AsyncResult fabricates a PENDING state and a TaskRevokedError
    result for unknown task ids, which used to leak as 200/500 responses.
    Gate every job_id-scoped route on the Redis metadata that the worker
    records when a job is queued."""
    if not get_job_meta(job_id):
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")


def _get_successful_job(job_id):
    _ensure_job_exists(job_id)
    job = worker.get_job(job_id)
    if job.status.lower() == "failed":
        raise HTTPException(status_code=400, detail="Job failed. Check the exception endpoint for more information")

    if not (job and job.is_finished):
        raise HTTPException(status_code=400, detail="Job is still running, try again later")

    return job


@router.get(
    "/{job_id}",
    summary="Check whether a job is done yet",
)
def get_job_status(job_id: str) -> str:
    """Return the current state of a queued job (``PENDING``, ``STARTED``, ``SUCCESS``, ``FAILURE``, ...).

    The canonical way to poll a background job started by any of the
    ``POST /v1/analytics/*`` or ``POST /v1/crud/*`` endpoints. 404 if the job id is
    unknown.
    """
    _ensure_job_exists(job_id)
    status: str = worker.get_job(job_id).status
    logger.info(f"status of job {job_id}: {status}")
    return status


@router.delete(
    "/{job_id}",
    summary="Forget a finished job",
)
def delete_job(job_id: str) -> dict:
    """Drop a finished or cancelled job from the tracker once you no longer need to see it in the listing — cancel it first if it's still running.

    Quirk: Celery treats unknown job ids as ``PENDING`` for safety, so passing an
    unknown id currently surfaces as 400 ("cannot delete a running job") rather than
    404. 404 only fires once metadata has actually been cleared. Cancel a running job
    via ``POST /{job_id}/cancel`` before deleting.
    """
    job = worker.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    job_status = job.status.lower()
    if job_status in ["pending", "started", "running"]:
        raise HTTPException(status_code=400, detail="Cannot delete a running job. Cancel it first.")

    result = redis.delete(f"job_meta:{job_id}")

    if result == 0:
        raise HTTPException(status_code=404, detail="Job not found")

    return {"message": f"Job {job_id} deleted successfully"}


@router.post(
    "/{job_id}/cancel",
    summary="Stop a running job",
)
def cancel_job(job_id: str) -> dict:
    """Revoke a queued or in-flight job so it stops eating worker cycles, e.g. when the user navigates away from a backtest creation flow or kills a slow prediction.

    400 if the job has already finished (``success`` / ``failure`` / ``revoked``) or is
    in an unexpected state; 404 if the id is unknown.
    """
    _ensure_job_exists(job_id)
    job = worker.get_job(job_id)
    job_status = job.status.lower()

    if job_status in ["success", "failure", "revoked"]:
        raise HTTPException(status_code=400, detail="Cannot cancel a job that has already finished or been cancelled")

    if job_status not in ["pending", "started", "running"]:
        raise HTTPException(status_code=400, detail=f"Cannot cancel job with status '{job.status}'")

    job.cancel()

    return {"message": f"Job {job_id} has been cancelled"}


@router.get(
    "/{job_id}/logs",
    summary="Read a job's captured logs",
)
def get_logs(job_id: str) -> str:
    """Tail the log output the worker captured while running this job — useful for debugging a failure or watching progress.

    404 if the job id is unknown; 400 if the log file hasn't been written yet (the job
    may not have started, or may be running on a worker that doesn't expose logs).
    """
    _ensure_job_exists(job_id)
    job = worker.get_job(job_id)
    logs: str = job.get_logs()
    if logs is None:
        raise HTTPException(status_code=400, detail=f"Log file not found for job ID '{job_id}'")
    return logs


@router.get(
    "/{job_id}/prediction_result",
    response_model=PredictionInfo,
    summary="Fetch the prediction a finished forecast job produced",
)
def get_prediction_result(
    job_id: str,
    session: Session = Depends(get_session),
):
    """Get the prediction row a successful forecast job wrote, in one hop instead of polling for status then looking up the id by hand.

    Useful right after the job's status flips to ``SUCCESS``. 400 if the job is still
    running or has failed; 404 if the job id or the resulting prediction is missing.
    """
    prediction_id = _get_successful_job(job_id).result
    prediction = session.get(Prediction, prediction_id)
    if prediction is None:
        raise HTTPException(status_code=404, detail=f"Prediction {prediction_id} not found")
    return prediction


@router.get(
    "/{job_id}/evaluation_result",
    response_model=BacktestRead,
    summary="Fetch the backtest a finished evaluation job produced",
)
def get_evaluation_result(
    job_id: str,
    session: Session = Depends(get_session),
):
    """Get the backtest a successful evaluation job wrote, in one hop instead of polling for status then looking up the id by hand.

    Useful right after the job's status flips to ``SUCCESS``. 400 if the job is still
    running or has failed; 404 if the job id or the resulting backtest is missing.
    """
    backtest_id = _get_successful_job(job_id).result
    backtest = session.get(Backtest, backtest_id)
    if backtest is None:
        raise HTTPException(status_code=404, detail=f"Backtest {backtest_id} not found")
    return backtest


class DataBaseResponse(BaseModel):
    id: int


@router.get(
    "/{job_id}/database_result",
    summary="Fetch the row id a generic finished job produced",
)
def get_database_result(job_id: str) -> DataBaseResponse:
    """Get the integer id a finished job wrote — used by job types whose only output is "a thing was created with id N" (most commonly dataset imports).

    Once you have the id, follow up with the appropriate CRUD endpoint to load the row.
    400 if the job is still running or has failed.
    """
    result = _get_successful_job(job_id).result
    return DataBaseResponse(id=result)


"""
Datasets for evaluation are versioned and stored
Datasets for predictions are not necessarily versioned and stored, but sent in each request
Evaluation should be run once, then using continous monitoring after
Task-id
"""


# todo: move this
class NaiveJob:
    def __init__(self, func, *args, **kwargs):
        # todo: init a root logger to capture all logs from the job
        self._exception_info = ""
        self._result = ""
        self._status = ""
        self._finished = False
        logger.info("Starting naive job")
        try:
            self._result = func(*args, **kwargs)
            self._status = "finished"
            logger.info("Naive job finished successfully")
            self._finished = True
        except Exception as e:
            self._exception_info = str(e)
            logger.info("Naive job failed with exception: %s", e)
            self._status = "failed"
            self._result = ""

    @property
    def id(self):
        return "naive_job"

    @property
    def status(self):
        return self._status

    @property
    def exception_info(self):
        return self._exception_info

    @property
    def progress(self):
        return 1

    @property
    def result(self):
        return self._result

    def cancel(self):
        pass

    @property
    def is_finished(self):
        return self._finished

    def get_logs(self, n_lines: int | None):
        """Retrives logs from the current job"""
        return ""


class NaiveWorker:
    job_class = NaiveJob

    def queue(self, func, *args, **kwargs):
        # return self.job_class(func(*args, **kwargs))
        return self.job_class(func, *args, **kwargs)
