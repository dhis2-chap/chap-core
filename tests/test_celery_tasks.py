import pytest

from chap_core.rest_api_src.celery_tasks import CeleryPool
from chap_core.util import redis_available


def time_consuming_function():
    import time
    time.sleep(3)


@pytest.mark.skipif(not redis_available(), reason="Redis not available")
@pytest.mark.celery(broker="redis://localhost:6379",
                    backend="redis://localhost:6379",
                    include=['chap_core.rest_api_src.celery_tasks'])
def test_list_jobs(celery_session_worker, big_request_json, test_config):
    pool = CeleryPool()
    job = pool.queue(time_consuming_function)
    jobs = pool.list_jobs()
    assert len(jobs) == 1
    assert jobs[0].id == job.id
    assert jobs[0].description.startswith("time_consuming_function")
