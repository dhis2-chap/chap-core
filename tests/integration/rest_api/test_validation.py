from fastapi.testclient import TestClient

from chap_core.rest_api.v1.rest_api import app

client = TestClient(app)


def test_(): ...
