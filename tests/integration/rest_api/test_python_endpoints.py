import pickle

import pytest

from chap_core.api_types import PredictionRequest
from chap_core.rest_api_src.v1.rest_api import predict, NaiveWorker, get_results, NaiveJob

from unittest.mock import patch

class PickleJob(NaiveJob):
    def __init__(self, result):
        self._result = pickle.loads(pickle.dumps(result))

class PickleWorker(NaiveWorker):
    job_class = PickleJob
    def queue(self, func, *args, **kwargs):
        args = [pickle.loads(pickle.dumps(arg)) for arg in args]
        kwargs = {k: pickle.loads(pickle.dumps(v)) for k, v in kwargs.items()}
        return super().queue(func, *args, **kwargs)

@pytest.mark.asyncio
@pytest.mark.skip
async def test_predict(big_request_json):
    big_request_json = PredictionRequest.model_validate_json(big_request_json)
    with patch("chap_core.rest_api.worker", PickleWorker()):
        await predict(big_request_json)
    results = await get_results()

    assert len(results['dataValues']) > 3
    assert isinstance((results['diseaseId']), str)