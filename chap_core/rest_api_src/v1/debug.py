import logging
from celery.result import AsyncResult
from fastapi import APIRouter, HTTPException
from ..celery_tasks import add_numbers, celery

router = APIRouter(prefix="/debug", tags=["debug"])
logger = logging.getLogger(__name__)

@router.get("/add-numbers")
def run_add_numbers(a: int, b: int):
    """Trigger a Celery task to add two numbers."""
    logger.info(f"Adding {a} and {b}")
    task = add_numbers.delay(a, b)
    return {"task_id": task.id, "status": "Task submitted"}


@router.get("/get-status")
def get_status(task_id: str) -> dict:
    """Get the status and result of a task."""
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
