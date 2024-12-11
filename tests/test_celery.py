import time
import pytest

from chap_core.api_types import RequestV2
from chap_core.rest_api_src.celery_tasks import celery_run, CeleryPool, add_numbers
from chap_core.rest_api_src.worker_functions import predict_pipeline_from_health_data, get_health_dataset
from  unittest.mock import patch
import logging
logging.basicConfig(level=logging.DEBUG)


def f(x, y):
    return x + y

# @pytest.mark.celery(broker="memory://",
#                     backend="cache+memory://", include=['chap_core.rest_api_src.v1.celery_tasks'])

@pytest.mark.celery(broker="redis://localhost:6379",
                    backend="redis://localhost:6379",
                    include=['chap_core.rest_api_src.celery_tasks'])
def test_add_numbers(celery_worker):
    job = celery_run.delay(add_numbers, 1, 2)
    time.sleep(2)
    assert job.state == "SUCCESS"
    assert job.result == 3
    celery_pool = CeleryPool()
    job = celery_pool.queue(f, 1, 2)
    # print(type(job._job))
    # print(type(job._result))
    time.sleep(2)
    # print(job.id)
    print(job.status)
    print(job.result)
    assert job.result == 3

@pytest.mark.celery(broker="redis://localhost:6379",
                    backend="redis://localhost:6379",
                    include=['chap_core.rest_api_src.celery_tasks'])
def test_predict_pipeline_from_health_data(celery_worker, big_request_json):
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


