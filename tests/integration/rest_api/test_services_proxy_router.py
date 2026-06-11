import fakeredis
import httpx
import pytest
from fastapi import FastAPI, Request, Response
from fastapi.testclient import TestClient

from chap_core.rest_api.app import app
from chap_core.rest_api.services.orchestrator import Orchestrator
from chap_core.rest_api.services.schemas import RegistrationRequest
from chap_core.rest_api.v2.dependencies import get_http_client, get_orchestrator

SERVICE_ID = "stub-model"
SERVICE_URL = "http://stub-service"


@pytest.fixture
def stub_app():
    """A minimal stand-in for a chapkit service that echoes what it received."""
    stub = FastAPI()

    @stub.api_route("/api/v1/echo/{rest:path}", methods=["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE"])
    async def echo(rest: str, request: Request):
        body = await request.body()
        return {
            "method": request.method,
            "path": rest,
            "query": dict(request.query_params),
            "body": body.decode() or None,
        }

    @stub.get("/api/v1/artifacts/$download")
    async def download():
        return Response(
            content=b"\x00\x01\x02chap",
            media_type="application/octet-stream",
            headers={"content-disposition": 'attachment; filename="artifact.bin"'},
        )

    @stub.get("/api/v1/multi-cookie")
    async def multi_cookie():
        response = Response(content=b"ok")
        response.headers.append("set-cookie", "a=1")
        response.headers.append("set-cookie", "b=2")
        return response

    return stub


@pytest.fixture
def test_orchestrator():
    orchestrator = Orchestrator(redis_client=fakeredis.FakeRedis())
    orchestrator.register(
        RegistrationRequest.model_validate(
            {
                "url": SERVICE_URL,
                "info": {
                    "id": SERVICE_ID,
                    "display_name": "Stub Model",
                    "model_metadata": {"author": "Test"},
                    "period_type": "monthly",
                },
            }
        )
    )
    return orchestrator


@pytest.fixture
def client(test_orchestrator, stub_app):
    stub_client = httpx.AsyncClient(transport=httpx.ASGITransport(app=stub_app))

    app.dependency_overrides[get_orchestrator] = lambda: test_orchestrator
    app.dependency_overrides[get_http_client] = lambda: stub_client
    yield TestClient(app, raise_server_exceptions=False)
    app.dependency_overrides.clear()


def test_get_passthrough(client):
    response = client.get(f"/v2/services/{SERVICE_ID}/run/api/v1/echo/here")

    assert response.status_code == 200
    data = response.json()
    assert data["method"] == "GET"
    assert data["path"] == "here"


def test_query_string_forwarded(client):
    response = client.get(f"/v2/services/{SERVICE_ID}/run/api/v1/echo/x?page=2&size=10")

    assert response.status_code == 200
    assert response.json()["query"] == {"page": "2", "size": "10"}


def test_head_request_allowed(client):
    response = client.head(f"/v2/services/{SERVICE_ID}/run/api/v1/echo/here")

    assert response.status_code == 200
    assert response.content == b""


def test_mutating_method_not_allowed(client):
    response = client.post(
        f"/v2/services/{SERVICE_ID}/run/api/v1/echo/configs",
        json={"name": "demo"},
    )

    assert response.status_code == 405


def test_repeated_response_headers_preserved(client):
    response = client.get(f"/v2/services/{SERVICE_ID}/run/api/v1/multi-cookie")

    assert response.status_code == 200
    assert response.headers.get_list("set-cookie") == ["a=1", "b=2"]


def test_binary_download_preserves_body_and_headers(client):
    response = client.get(f"/v2/services/{SERVICE_ID}/run/api/v1/artifacts/$download")

    assert response.status_code == 200
    assert response.content == b"\x00\x01\x02chap"
    assert response.headers["content-type"] == "application/octet-stream"
    assert response.headers["content-disposition"] == 'attachment; filename="artifact.bin"'


def test_unknown_service_returns_404(client):
    response = client.get("/v2/services/does-not-exist/run/api/v1/echo/x")

    assert response.status_code == 404


def test_unreachable_service_returns_502(test_orchestrator):
    def raise_connect_error(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    dead_client = httpx.AsyncClient(transport=httpx.MockTransport(raise_connect_error))

    app.dependency_overrides[get_orchestrator] = lambda: test_orchestrator
    app.dependency_overrides[get_http_client] = lambda: dead_client
    try:
        test_client = TestClient(app, raise_server_exceptions=False)
        response = test_client.get(f"/v2/services/{SERVICE_ID}/run/api/v1/echo/x")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 502
