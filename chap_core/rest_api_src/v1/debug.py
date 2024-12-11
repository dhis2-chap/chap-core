import logging
from typing import Optional

from celery.result import AsyncResult
from fastapi import APIRouter, HTTPException
from ..celery_tasks import celery, CeleryPool

router = APIRouter(prefix="/debug", tags=["debug"])
logger = logging.getLogger(__name__)
cur_job = None
celery_pool = CeleryPool()


@router.get("/add-numbers")
def run_add_numbers(a: int, b: int):
    """Trigger a Celery task to add two numbers."""
    global cur_job
    logger.info(f"Adding {a} and {b}")
    # task = add_numbers.delay(a, b)
    # job= celery_pool._celery.send_task("celery_tasks.add_numbers", args=[a, b])
    # cur_job = celery_pool.queue(_add_numbers, a, b)
    return None
    return {"task_id": cur_job.id, "status": "Task submitted"}


@router.get("/get-status")
def get_status(task_id: Optional[str] = None) -> dict:
    """Get the status and result of a task."""
    task_id = task_id or cur_job.id
    task_result = AsyncResult(task_id, app=celery)

    # Check if task is in a valid state
    logger.info(f"Task {task_id}: {task_result}")
    if task_result.state == "PENDING":
        raise HTTPException(status_code=404, detail="Task not found or still pending execution.")

    result = {
        "task_id": task_id,
        "status": task_result.state,
        "result": task_result.result if task_result.successful() else '',
        "error": str(task_result.result) if task_result.failed() else '',
    }

    return result
