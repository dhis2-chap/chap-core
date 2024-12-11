import os
from typing import Callable, Generic

from celery import Celery
from celery.result import AsyncResult
from dotenv import find_dotenv, load_dotenv

from .worker_functions import predict_pipeline_from_health_data, evaluate
import celery

from ..worker.interface import ReturnType

# predict_pipeline_from_health_data = celery.task(predict_pipeline_from_health_data)
celery = Celery(
    "worker",
    broker="redis://redis:6379",
    backend="redis://redis:6379"
)
celery.conf.update(
    task_serializer="pickle",
    accept_content=["pickle"],  # Allow pickle serialization
    result_serializer="pickle",
)

def _add_numbers(a: int, b: int):
    return a + b


#add_numbers = celery.task()(_add_numbers)
predict_pipeline_from_health_data = celery.task(predict_pipeline_from_health_data)
evaluate = celery.task(evaluate)


class CeleryJob(Generic[ReturnType]):
    """Wrapper for a Celery Job"""

    def __init__(self, job: celery.Task, app: Celery):
        self._job = job
        self._app = app

    @property
    def _result(self):
        return AsyncResult(self._job.id, app=celery)

    @property
    def status(self) -> str:
        return self._result.state

    @property
    def result(self) -> ReturnType:
        return self._result.result

    @property
    def progress(self) -> float:
        return 0

    def cancel(self):
        self._result.revoke()

    @property
    def id(self):
        return self._job.id

    @property
    def is_finished(self) -> bool:
        return self._result.state in ("SUCCESS", "FAILURE")


@celery.task(name="celery_tasks.celery_run")
def celery_run(func, *args, **kwargs):
    return func(*args, **kwargs)


class CeleryPool(Generic[ReturnType]):
    """Simple abstraction for a Celery Worker"""

    def __init__(self):
        host, port = self.read_environment_variables()
        self._celery = Celery(
            "worker",
            broker=f"redis://{host}:{port}",
            backend=f"redis://{host}:{port}",
        )

    def read_environment_variables(self):
        load_dotenv(find_dotenv())
        host = os.environ.get("REDIS_HOST")
        port = os.environ.get("REDIS_PORT")

        # using default values if environment variables are not set
        if host is None:
            host = "localhost"
        if port is None:
            port = "6379"

        return host, port

    def queue(self, func: Callable[..., ReturnType], *args, **kwargs) -> CeleryJob[ReturnType]:
        #task = predict_pipeline_from_health_data if func.__name__ == "predict_pipeline_from_health_data" else celery.task()(func)
        #task = task.delay(*args, **kwargs)
        job = celery_run.delay(func, *args, **kwargs)
        return CeleryJob(job, app=self._celery)

    def get_job(self, task_id: str) -> CeleryJob[ReturnType]:
        return CeleryJob(AsyncResult(task_id, app=self._celery), app=self._celery)

    def __del__(self):
        self._celery.control.revoke()
