import time
import pytest

from chap_core.api_types import RequestV2
from chap_core.rest_api_src.celery_tasks import celery_run, CeleryPool, add_numbers
from chap_core.rest_api_src.worker_functions import predict_pipeline_from_health_data, get_health_dataset
from  unittest.mock import patch
import logging
from celery import Celery
from chap_core.util import redis_available

# logging.basicConfig(level=logging.DEBUG)


def f(x, y):
    return x + y

# @pytest.mark.celery(broker="memory://",
#                     backend="cache+memory://", include=['chap_core.rest_api_src.v1.celery_tasks'])

@pytest.mark.skipif(not redis_available(), reason="Redis not available")
@pytest.mark.celery(broker="redis://localhost:6379",
                    backend="redis://localhost:6379",
                    include=['chap_core.rest_api_src.celery_tasks'])
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
    # print(type(job._job))
    # print(type(job._result))
    time.sleep(2)
    # print(job.id)
    print(job.status)
    print(job.result)
    assert job.result == 3


@pytest.mark.skipif(not redis_available(), reason="Redis not available")
@pytest.mark.celery(broker="redis://localhost:6379",
                    backend="redis://localhost:6379",
                    include=['chap_core.rest_api_src.celery_tasks'])
def test_predict_pipeline_from_health_data(celery_session_worker, big_request_json):
    data = RequestV2.model_validate_json(big_request_json)
    health_data = get_health_dataset(data).model_dump()
    job = celery_run.delay(predict_pipeline_from_health_data, health_data, 'naive_model', 2, 'disease')
    for i in range(30):
        time.sleep(2)
        if job.state == "SUCCESS":
            break
        if job.state == "FAILURE":
            assert False
    assert job.state == "SUCCESS"


def function_with_logging(a, b):
    logger = logging.getLogger(__name__)
    logger.info("Some testlogging")
    return add_numbers(a, b)


@pytest.mark.skipif(not redis_available(), reason="Redis not available")
@pytest.mark.celery(broker="redis://localhost:6379",
                    backend="redis://localhost:6379",
                    include=['chap_core.rest_api_src.celery_tasks'])
@pytest.mark.skip(reason="Not stable")
def test_celery_logging(celery_session_worker):
    pool = CeleryPool()
    job = pool.queue(function_with_logging, 1, 2)
    #job = celery_run.delay(add_numbers, 1, 2)
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

