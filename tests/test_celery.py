import inspect
import logging
import time
from types import SimpleNamespace

import pytest
from kombu.serialization import dumps, loads

from chap_core.api_types import PredictionRequest
from chap_core.assessment.weather_providers import DEFAULT_WEATHER_PROVIDER_ID
from chap_core.database.dataset_tables import DataSetCreateInfo, DataSource, MaskedRange
from chap_core.log_config import get_status_logger
from chap_core.rest_api import celery_tasks
from chap_core.rest_api import db_worker_functions as wf
from chap_core.rest_api.celery_tasks import JOB_NAME_KW, JOB_TYPE_KW, CeleryPool, add_numbers, celery_run
from chap_core.rest_api.data_models import BacktestCreate, PredictionParams
from chap_core.rest_api.job_registry import job_function, job_name_for, json_args, json_kwargs, resolve_job
from chap_core.rest_api.worker_functions import WorkerConfig, get_health_dataset
from chap_core.util import redis_available


@job_function
def f(x, y):
    return x + y


@pytest.mark.skipif(not redis_available(), reason="Redis not available")
@pytest.mark.celery(
    broker="redis://localhost:6379", backend="redis://localhost:6379", include=["chap_core.rest_api.celery_tasks"]
)
@pytest.mark.slow
def test_add_numbers(celery_session_worker):
    job = celery_run.delay(job_name_for(add_numbers), 1, 2)
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


@job_function
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


@job_function
def time_consuming_function():
    import time

    time.sleep(3)


@job_function
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
        args=(job_name_for(function_with_prediction_setup_id),),
        kwargs={
            JOB_TYPE_KW: "create_prediction",
            JOB_NAME_KW: "prediction job",
            "prediction_setup_id": 12,
        },
    )

    assert fake_redis.hsets[-1][1]["prediction_setup_id"] == "12"


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


@job_function
def function_with_status_logging():
    get_status_logger().info("progress message")
    logging.getLogger("some.library").warning("library detail")


def test_tracked_task_writes_status_and_library_logs_to_task_log(monkeypatch, tmp_path):
    monkeypatch.setattr(celery_tasks, "r", _FakeRedis())
    monkeypatch.setattr(celery_tasks, "CHAP_LOGS_DIR", tmp_path)
    monkeypatch.setitem(celery_tasks.app.conf, "task_always_eager", True)

    result = celery_tasks.celery_run.apply_async(args=(job_name_for(function_with_status_logging),))

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
    assert unmatched_job.model_id is None
    assert unmatched_job.model_version is None
    assert unmatched_job.dataset_id is None
    assert unmatched_job.parameters == {}


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
    # Redis is shared with other test processes under xdist, so look up our own job
    # rather than assuming it is the most recent one.
    matching = [j for j in jobs if j.id == job.id]
    assert matching, "Job not found in list of jobs"
    assert matching[0].type == "time_consuming_function"
    assert matching[0].name == "test_job_name"


def test_queue_refuses_an_unregistered_function():
    def not_registered():
        return 1

    with pytest.raises(ValueError, match="not registered"):
        CeleryPool().queue(not_registered)


def test_worker_refuses_an_unknown_job_name(monkeypatch, tmp_path):
    monkeypatch.setattr(celery_tasks, "r", _FakeRedis())
    monkeypatch.setattr(celery_tasks, "CHAP_LOGS_DIR", tmp_path)
    monkeypatch.setitem(celery_tasks.app.conf, "task_always_eager", True)

    result = celery_run.apply_async(args=("os.system", "echo pwned"))

    assert result.status == "FAILURE"
    assert isinstance(result.result, ValueError)


def test_pydantic_arguments_survive_the_json_broker():
    info = BacktestCreate(name="backtest", dataset_id=1, model_id=2)
    params = PredictionParams(model_id="model", n_periods=3, future_weather_provider=DEFAULT_WEATHER_PROVIDER_ID)
    sources = [DataSource(covariate="rainfall", data_element_id="abc")]
    config = WorkerConfig(is_test=True, failing_services=("climate",))

    def over_the_broker(func, *args, **kwargs):
        content_type, encoding, body = dumps(
            (job_name_for(func), json_args(args), json_kwargs(kwargs)), serializer="json"
        )
        name, sent_args, sent_kwargs = loads(body, content_type, encoding)
        _, args, kwargs = resolve_job(name, tuple(sent_args), sent_kwargs)
        return inspect.signature(func).bind_partial(*args, **kwargs).arguments

    assert over_the_broker(wf.run_backtest, info)["info"] == info
    received = over_the_broker(
        wf.predict_pipeline_from_composite_dataset,
        ["rainfall"],
        {"data_dict": {}, "polygons": None},
        "prediction",
        dataset_create_info={"name": "dataset", "type": "prediction"},
        prediction_params=params,
        worker_config=config,
    )
    assert received["prediction_params"] == params
    assert received["dataset_create_info"] == DataSetCreateInfo(name="dataset", type="prediction")
    assert received["worker_config"] == config
    masked = [MaskedRange(start="202006", end="202008", name="covid")]
    received = over_the_broker(
        wf.harmonize_and_add_dataset,
        ["rainfall"],
        [],
        {"data_dict": {}},
        "dataset",
        "evaluation",
        data_sources=sources,
        masked_ranges=masked,
    )
    assert received["data_sources"] == sources
    assert received["masked_ranges"] == masked


@job_function
def database_of(session):
    return session.session.get_bind().url.render_as_string(hide_password=False)


def test_worker_ignores_a_database_url_in_the_message(monkeypatch, tmp_path):
    monkeypatch.setattr(celery_tasks, "r", _FakeRedis())
    monkeypatch.setattr(celery_tasks, "CHAP_LOGS_DIR", tmp_path)
    monkeypatch.setitem(celery_tasks.app.conf, "task_always_eager", True)
    own_database = f"sqlite:///{tmp_path}/worker.db"
    monkeypatch.setenv("CHAP_DATABASE_URL", own_database)

    result = celery_tasks.celery_run_with_session.apply_async(
        args=(job_name_for(database_of),), kwargs={"database_url": "postgresql://attacker@evil.example/chap"}
    )

    assert result.result == own_database


def test_worker_without_a_database_url_refuses_the_job(monkeypatch, tmp_path):
    monkeypatch.setattr(celery_tasks, "r", _FakeRedis())
    monkeypatch.setattr(celery_tasks, "CHAP_LOGS_DIR", tmp_path)
    monkeypatch.setitem(celery_tasks.app.conf, "task_always_eager", True)
    monkeypatch.delenv("CHAP_DATABASE_URL", raising=False)

    result = celery_tasks.celery_run_with_session.apply_async(
        args=(job_name_for(database_of),), kwargs={"database_url": "postgresql://attacker@evil.example/chap"}
    )

    assert result.status == "FAILURE"
    assert isinstance(result.result, RuntimeError)


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("redis://:s3cret@redis.internal:6379/0", "redis.internal:6379"),
        ("redis://user:s3cret@redis.internal/0", "redis.internal"),
        ("redis://localhost:6379", "localhost:6379"),
        ("redis://:s3cret@h1:6379;redis://:s3cret@h2:6379", "h1"),
    ],
)
def test_broker_host_drops_credentials(url, expected):
    assert celery_tasks.broker_host(url) == expected
