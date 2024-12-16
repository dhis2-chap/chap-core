import json
import time
import unittest

import pytest
import logging
from chap_core.api_types import PredictionRequest
from chap_core.rest_api_src.v1.rest_api import app
from fastapi.testclient import TestClient

from chap_core.util import redis_available
logger = logging.getLogger(__name__)
client = TestClient(app)
base_path = '/v1/jobs'
evaluate_path = "/v1/evaluate"

@pytest.mark.skipif(not redis_available(), reason="Redis not available")
@pytest.mark.celery(broker="redis://localhost:6379",
                    backend="redis://localhost:6379",
                    include=['chap_core.rest_api_src.celery_tasks'])
def test_predict(big_request_json, celery_session_worker, gee_mock):
    data = json.loads(big_request_json)
    data['estimator_id'] = "naive_model"
    with unittest.mock.patch("chap_core.rest_api_src.worker_functions.Era5LandGoogleEarthEngine", gee_mock):
        response = client.post(evaluate_path, json=data)
    assert response.status_code == 200, response.json()
    #response = client.post(evaluate_path, json=data)
    print(response.json())
    task_id = response.json()['task_id']

    for _ in range(200):
        status = client.get(f"{base_path}/{task_id}")
        assert status.status_code == 200, status.json()
        if status.json() not in ['PENDING', 'STARTED']:
            break
        logger.info(status.json())
        time.sleep(1)
    assert status.status_code == 200, status.json()
    assert status.json() == 'SUCCESS'
    result = client.get(f"{base_path}/{task_id}/evaluation_result")
    assert result.status_code == 200, result.json()
    print(result.json())


