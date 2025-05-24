import os
from datetime import datetime
from pathlib import Path
from typing import Callable, Generic
import logging
from celery import Celery, shared_task, Task
from celery.result import AsyncResult
from redis import Redis
from dotenv import find_dotenv, load_dotenv
import json

import celery
from pydantic import BaseModel
from sqlalchemy import create_engine

from ..database.database import SessionWrapper
from ..worker.interface import ReturnType
from celery.utils.log import get_task_logger

# We use get_task_logger to ensure we get the Celery-friendly logger
# but you could also just use logging.getLogger(__name__) if you prefer.
logger = get_task_logger(__name__)
logger.setLevel(logging.INFO)


# Send database url in function queue call. Have a dict in module of database url to engines. Look up engine in dict
class JobDescription(BaseModel):
    id: str
    type: str
    name: str
    status: str
    start_time: str | None
    end_time: str | None
    result: str | None


def read_environment_variables():
    load_dotenv(find_dotenv())
    host = os.getenv("CELERY_BROKER", "redis://localhost:6379")
    return host


# Setup celery
url = read_environment_variables()
logger.info(f"Connecting to {url}")
app = Celery("worker", broker=url, backend=url)
app.conf.update(
    task_serializer="pickle",
    accept_content=["pickle"],  # Allow pickle serialization
    result_serializer="pickle",
    # Enables tracking of job lifecycle
    task_track_started=True,
    task_send_sent_event=True,
    worker_send_task_events=True,
)


# Setup Redis connection (for job metadata)
# TODO: switch to using utils.load_redis()?
redis_url = "redis" if "localhost" not in url else "localhost"
r = Redis(host=redis_url, port=6379, db=2, decode_responses=True)  # TODO: how to set this better?


# logger.warning("No database URL set")
# This is hacky, but defaults to using the test database. Should be synched with what is setup in conftest
# engine = create_engine("sqlite:///test.db", connect_args={"check_same_thread": False})


class TrackedTask(Task):
    def __call__(self, *args, **kwargs):
        # Extract the current task id
        task_id = self.request.id

        # Create a file handler for this task's logs
        file_handler = logging.FileHandler(Path("logs") / f"task_{task_id}.txt")
        file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        file_handler.setFormatter(file_formatter)

        # Remember old handlers so we can restore them later
        old_handlers = logger.handlers[:]

        # also add this handler to the root-logger, so that logging done by other packages is also logged
        root_logger = logging.getLogger()
        old_root_handlers = root_logger.handlers[:]
        root_logger.addHandler(file_handler)

        # Replace the logger handlers with our per-task file handler
        logger.handlers = [file_handler]
        # also add stdout handler
        logger.addHandler(logging.StreamHandler())

        try:
            # Mark as started when the task is actually executing
            r.hmset(
                f"job_meta:{task_id}",
                {
                    "status": "STARTED",
                    # "start_time": datetime.now().isoformat(), # update the start time
                },
            )
            # Execute the actual task
            return super().__call__(*args, **kwargs)

        finally:
            # Close the file handler and restore old handlers after the task is done
            file_handler.close()
            logger.handlers = old_handlers
            root_logger.handlers = old_root_handlers

    def apply_async(self, args=None, kwargs=None, **options):
        # print('apply async', args, kwargs, options)
        job_name = kwargs.pop(JOB_NAME_KW, None) or "Unnamed"
        job_type = kwargs.pop(JOB_TYPE_KW, None) or "Unspecified"
        result = super().apply_async(args=args, kwargs=kwargs, **options)

        r.hmset(
            f"job_meta:{result.id}",
            {"job_name": job_name, "job_type": job_type, "status": "PENDING", "start_time": datetime.now().isoformat()},
        )

        return result

    def on_success(self, retval, task_id, args, kwargs):
        print("success!")
        # start = float(r.hget(f"job_meta:{task_id}", "start_time") or time.time())
        # duration = time.time() - start

        r.hmset(
            f"job_meta:{task_id}",
            {
                "status": "SUCCESS",
                # "duration": duration,
                "result": json.dumps(retval),
                "end_time": datetime.now().isoformat(),
            },
        )

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        print("failure!")
        # start = float(r.hget(f"job_meta:{task_id}", "start_time") or time.time())
        # duration = time.time() - start

        r.hmset(
            f"job_meta:{task_id}",
            {
                "status": "FAILURE",
                # "duration": duration,
                "error": str(exc),
                "traceback": str(einfo.traceback),
                "end_time": datetime.now().isoformat(),
            },
        )


@shared_task(name="celery.ping")
def ping():
    return "pong"


def add_numbers(a: int, b: int):
    logger.info(f"Adding {a} + {b}")
    return a + b


# set base to TrackedTask to enable per-task logging
@app.task(base=TrackedTask)
def celery_run(func, *args, **kwargs):
    return func(*args, **kwargs)


ENGINES_CACHE = {}


@app.task(base=TrackedTask)
def celery_run_with_session(func, *args, **kwargs):
    database_url = kwargs.pop("database_url")
    if database_url not in ENGINES_CACHE:
        ENGINES_CACHE[database_url] = create_engine(database_url)
    engine = ENGINES_CACHE[database_url]
    with SessionWrapper(engine) as session:
        return func(*args, **kwargs | {"session": session})


JOB_TYPE_KW = "__job_type__"
JOB_NAME_KW = "__job_name__"


class CeleryJob(Generic[ReturnType]):
    """Wrapper for a Celery Job"""

    def __init__(self, job: celery.Task, app: Celery):
        self._job = job
        self._app = app

    @property
    def _result(self) -> AsyncResult:
        return AsyncResult(self._job.id, app=self._app)

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
        self._result.revoke(terminate=True)
        
        # Update Redis metadata to reflect the cancellation
        r.hmset(
            f"job_meta:{self._job.id}",
            {
                "status": "REVOKED",
                "end_time": datetime.now().isoformat(),
            },
        )

    @property
    def id(self):
        return self._job.id

    @property
    def is_finished(self) -> bool:
        return self._result.state in ("SUCCESS", "FAILURE", "REVOKED")

    @property
    def exception_info(self) -> str:
        return str(self._result.traceback or "")

    def get_logs(self) -> str:
        log_file = Path("app/logs") / f"task_{self._job.id}.txt"  # TODO: not sure why have to specify app/logs...
        if log_file.exists():
            logs = log_file.read_text()
            job_meta = get_job_meta(self.id)
            if job_meta["status"] == "FAILURE":
                logs += "\n" + job_meta["traceback"]
            return logs
        return None


class CeleryPool(Generic[ReturnType]):
    """Simple abstraction for a Celery Worker"""

    def __init__(self, celery: Celery = None):
        assert celery is None
        self._celery = app

    def queue(self, func: Callable[..., ReturnType], *args, **kwargs) -> CeleryJob[ReturnType]:
        job = celery_run.delay(func, *args, **kwargs)
        return CeleryJob(job, app=self._celery)

    def queue_db(self, func: Callable[..., ReturnType], *args, **kwargs) -> CeleryJob[ReturnType]:
        job = celery_run_with_session.delay(func, *args, **kwargs)
        return CeleryJob(job, app=self._celery)

    def get_job(self, task_id: str) -> CeleryJob[ReturnType]:
        return CeleryJob(AsyncResult(task_id, app=self._celery), app=self._celery)

    # def _describe_job(self, job_info: dict) -> str:
    #    func, *args = job_info["args"]
    #    func_name = func.__name__
    #    return f"{func_name}({', '.join(map(str, args))})"

    # def list_jobs(self) -> List[JobDescription]:
    #     all_jobs = {'active': self._celery.control.inspect().active(),}
    #                 #'scheduled': self._celery.control.inspect().scheduled(),
    #                 #'reserved': self._celery.control.inspect().reserved()}
    #     print(all_jobs)

    #     return [JobDescription(id=info['id'],
    #                            description=self._describe_job(info),
    #                            status=status,
    #                            start_time=datetime.fromtimestamp(info["time_start"]),
    #                            hostname=hostname,
    #                            type=self._get_job_type(info))
    #             for status, host_dict in all_jobs.items() for hostname, jobs in host_dict.items() for info in jobs]

    def list_jobs(self, status: str = None):
        """List all tracked jobs stored by Redis. Optional filter by status: PENDING, STARTED, SUCCESS, FAILURE, REVOKED"""
        keys = r.keys("job_meta:*")
        jobs = []

        for key in keys:
            task_id = key.split(":")[1]
            meta = r.hgetall(key)
            meta["task_id"] = task_id
            if status is None or meta.get("status") == status:
                jobs.append(meta)

        return [
            JobDescription(
                id=meta["task_id"],
                type=meta.get("job_type", "Unspecified"),
                name=meta.get("job_name", "Unnamed"),
                status=meta["status"],
                start_time=meta.get("start_time", None),
                end_time=meta.get("end_time", None),
                result=meta.get("result", None),
            )
            for meta in sorted(jobs, key=lambda x: x.get("start_time", datetime(1900, 1, 1).isoformat()), reverse=True)
        ]


def get_job_meta(task_id: str):
    """Fetch Redis metadata for a job by task ID."""
    key = f"job_meta:{task_id}"
    if not r.exists(key):
        return None
    return r.hgetall(key)
