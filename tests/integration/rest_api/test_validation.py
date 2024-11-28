from chap_core.rest_api_src.v1.rest_api import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_():
    ...