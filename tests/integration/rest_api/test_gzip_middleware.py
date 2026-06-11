from fastapi.testclient import TestClient

from chap_core.rest_api.app import app


class TestGZipMiddleware:
    def test_large_response_is_gzipped(self):
        client = TestClient(app)
        response = client.get("/openapi.json", headers={"Accept-Encoding": "gzip"})

        assert response.status_code == 200
        assert response.headers.get("content-encoding") == "gzip"

    def test_response_not_gzipped_when_client_does_not_accept(self):
        client = TestClient(app)
        response = client.get("/openapi.json", headers={"Accept-Encoding": "identity"})

        assert response.status_code == 200
        assert response.headers.get("content-encoding") != "gzip"

    def test_small_response_not_gzipped(self):
        client = TestClient(app)
        response = client.get("/health", headers={"Accept-Encoding": "gzip"})

        assert response.status_code == 200
        assert response.headers.get("content-encoding") != "gzip"
