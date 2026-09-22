import logging
import time
from types import SimpleNamespace

import pytest

from chap_core.api_types import PredictionRequest
from chap_core.log_config import get_status_logger
from chap_core.rest_api import celery_tasks
from chap_core.rest_api.celery_tasks import JOB_NAME_KW, JOB_TYPE_KW, CeleryPool, add_numbers, celery_run
from chap_core.rest_api.worker_functions import get_health_dataset
from chap_core.util import redis_available


def f(x, y):
    return x + y


@pytest.mark.skipif(not redis_available(), reason="Redis not available")
@pytest.mark.celery(
    broker="redis://localhost:6379", backend="redis://localhost:6379", include=["chap_core.rest_api.celery_tasks"]
)
@pytest.mark.slow
def test_add_numbers(celery_session_worker):
    job = celery_run.delay(add_numbers, 1, 2)
    time.sleep(2)
    assert job.state == "SUCCESS"
    assert job.result == 3
    celery_pool = CeleryPool()
    job = celery_pool.queue(f, 1, 2)
    while job.status != "SUCCESS":
        time.sleep(2)
        if job.status == "FAILURE":
            assert False, "Job failed"
    time.sleep(2)
    print(job.status)
    print(job.result)
    assert job.result == 3


def function_with_logging(a, b):
    logger = logging.getLogger(__name__)
    logger.info("Some testlogging")
    return add_numbers(a, b)


@pytest.mark.skipif(not redis_available(), reason="Redis not available")
@pytest.mark.celery(
    broker="redis://localhost:6379", backend="redis://localhost:6379", include=["chap_core.rest_api.celery_tasks"]
)
@pytest.mark.skip(reason="Not stable")
def test_celery_logging(celery_session_worker):
    pool = CeleryPool()
    job = pool.queue(function_with_logging, 1, 2)
    job_id = job.id
    n_tries = 0
    while job.status != "SUCCESS":
        time.sleep(2)

        if job.status == "FAILURE" or n_tries > 10:
            assert False, "Job failed"

        n_tries += 1

    job2 = pool.get_job(job_id)
    assert job2.result == 3

    logs = job2.get_logs()
    assert "Some testlogging" in logs


def time_consuming_function():
    import time

    time.sleep(3)


def function_with_prediction_setup_id(prediction_setup_id: int | None = None):
    return prediction_setup_id


class _FakeRedis:
    def __init__(self):
        self.hsets: list[tuple[str, dict]] = []

    def hset(self, key, mapping):
        self.hsets.append((key, dict(mapping)))

    def keys(self, _pattern):
        return ["job_meta:job-1", "job_meta:job-2"]

    def hgetall(self, key):
        return {
            "job_meta:job-1": {
                "job_name": "prediction one",
                "job_type": "create_prediction",
                "status": "PENDING",
                "start_time": "2026-01-01T00:00:00",
                "prediction_setup_id": "12",
            },
            "job_meta:job-2": {
                "job_name": "prediction two",
                "job_type": "create_prediction",
                "status": "PENDING",
                "start_time": "2026-01-01T00:00:01",
            },
        }[key]


def test_apply_async_stores_prediction_setup_id_in_job_metadata(monkeypatch, tmp_path):
    fake_redis = _FakeRedis()
    monkeypatch.setattr(celery_tasks, "r", fake_redis)
    monkeypatch.setattr(celery_tasks, "CHAP_LOGS_DIR", tmp_path)
    monkeypatch.setitem(celery_tasks.app.conf, "task_always_eager", True)

    celery_tasks.celery_run.apply_async(
        args=(function_with_prediction_setup_id,),
        kwargs={
            JOB_TYPE_KW: "create_prediction",
            JOB_NAME_KW: "prediction job",
            "prediction_setup_id": 12,
        },
    )

    assert fake_redis.hsets[0][1]["prediction_setup_id"] == "12"


class _NoMetaRedis:
    def exists(self, _key):
        return 0


def test_get_logs_returns_debug_log(monkeypatch, tmp_path):
    monkeypatch.setattr(celery_tasks, "CHAP_LOGS_DIR", tmp_path)
    monkeypatch.setattr(celery_tasks, "r", _NoMetaRedis())
    (tmp_path / "task_job-1.debug.txt").write_text("progress line\nlibrary detail\n")

    job = celery_tasks.CeleryJob(SimpleNamespace(id="job-1"), app=celery_tasks.app)

    assert job.get_logs() == "progress line\nlibrary detail\n"


def test_get_logs_returns_bounded_tail_of_large_log(monkeypatch, tmp_path):
    monkeypatch.setattr(celery_tasks, "CHAP_LOGS_DIR", tmp_path)
    monkeypatch.setattr(celery_tasks, "r", _NoMetaRedis())
    monkeypatch.setattr(celery_tasks, "LOG_TAIL_BYTES", 40)
    (tmp_path / "task_job-1.debug.txt").write_text("first line is long enough to be cut\nsecond line\nlast line\n")

    logs = celery_tasks.CeleryJob(SimpleNamespace(id="job-1"), app=celery_tasks.app).get_logs()

    assert logs.startswith("[... earlier log output truncated ...]\n")
    assert "first line" not in logs
    assert logs.endswith("second line\nlast line\n")


def function_with_status_logging():
    get_status_logger().info("progress message")
    logging.getLogger("some.library").warning("library detail")


def test_tracked_task_writes_status_and_library_logs_to_task_log(monkeypatch, tmp_path):
    monkeypatch.setattr(celery_tasks, "r", _FakeRedis())
    monkeypatch.setattr(celery_tasks, "CHAP_LOGS_DIR", tmp_path)
    monkeypatch.setitem(celery_tasks.app.conf, "task_always_eager", True)

    result = celery_tasks.celery_run.apply_async(args=(function_with_status_logging,))

    log = (tmp_path / f"task_{result.id}.debug.txt").read_text()
    assert "progress message" in log
    assert "library detail" in log


def test_list_jobs_includes_prediction_setup_id(monkeypatch):
    fake_redis = _FakeRedis()
    monkeypatch.setattr(celery_tasks, "r", fake_redis)

    jobs = CeleryPool().list_jobs()

    matching_job = next(job for job in jobs if job.id == "job-1")
    assert matching_job.prediction_setup_id == 12
    unmatched_job = next(job for job in jobs if job.id == "job-2")
    assert unmatched_job.prediction_setup_id is None


@pytest.mark.skipif(not redis_available(), reason="Redis not available")
@pytest.mark.celery(
    broker="redis://localhost:6379", backend="redis://localhost:6379", include=["chap_core.rest_api.celery_tasks"]
)
def test_list_jobs(celery_session_worker, big_request_json, test_config):
    pool = CeleryPool()
    job = pool.queue(time_consuming_function, **{JOB_TYPE_KW: "time_consuming_function", JOB_NAME_KW: "test_job_name"})
    time.sleep(2)
    jobs = pool.list_jobs()
    assert len(jobs) >= 1
    assert any(j.id == job.id for j in jobs), "Job not found in list of jobs"
    # assert jobs[0].id == job.id
    assert jobs[0].type == "time_consuming_function"
    assert jobs[0].name == "test_job_name"


def fail_with_original_request(task_id):
    import json

    assert json.loads(celery_tasks.r.hget(f"job_meta:{task_id}", "request")) == {"name": "original"}
    raise RuntimeError("deliberate failure")


def test_original_request_survives_worker_failure(monkeypatch, tmp_path):
    import json
    import fakeredis

    store = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(celery_tasks, "r", store)
    monkeypatch.setattr(celery_tasks, "CHAP_LOGS_DIR", tmp_path)
    monkeypatch.setitem(celery_tasks.app.conf, "task_always_eager", True)
    result = celery_tasks.celery_run.apply_async(
        args=(fail_with_original_request, "failed-request"),
        kwargs={celery_tasks.JOB_REQUEST_KW: {"name": "original"}},
        task_id="failed-request",
    )
    assert result.state == "FAILURE"
    assert str(result.result) == "deliberate failure"
    metadata = store.hgetall("job_meta:failed-request")
    assert metadata["status"] == "FAILURE"
    assert json.loads(metadata["request"]) == {"name": "original"}


def test_failed_dispatch_removes_request(monkeypatch):
    import fakeredis
    from celery import Task

    store = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(celery_tasks, "r", store)

    def reject_dispatch(*args, **kwargs):
        assert store.hexists("job_meta:not-queued", "request")
        raise RuntimeError("broker unavailable")

    monkeypatch.setattr(Task, "apply_async", reject_dispatch)
    with pytest.raises(RuntimeError, match="broker unavailable"):
        celery_tasks.celery_run.apply_async(
            args=(add_numbers, 1, 2),
            kwargs={celery_tasks.JOB_REQUEST_KW: {"name": "original"}},
            task_id="not-queued",
        )
    assert not store.exists("job_meta:not-queued")
