from fastapi.testclient import TestClient

from chap_core.rest_api.app import CORS_ORIGINS_ENV_VAR, app, get_cors_origins


def preflight(origin: str):
    client = TestClient(app)
    return client.options(
        "/health",
        headers={"Origin": origin, "Access-Control-Request-Method": "GET"},
    )


class TestCorsMiddleware:
    def test_refuses_an_unknown_origin(self):
        response = preflight("https://evil.example")

        assert response.status_code == 400
        assert "access-control-allow-origin" not in response.headers


class TestGetCorsOrigins:
    def test_defaults_to_the_local_dev_server(self, monkeypatch):
        monkeypatch.delenv(CORS_ORIGINS_ENV_VAR, raising=False)

        assert get_cors_origins() == ["http://localhost:3000"]

    def test_empty_value_falls_back_to_the_default(self, monkeypatch):
        monkeypatch.setenv(CORS_ORIGINS_ENV_VAR, "")

        assert get_cors_origins() == ["http://localhost:3000"]

    def test_reads_comma_separated_origins(self, monkeypatch):
        monkeypatch.setenv(CORS_ORIGINS_ENV_VAR, "https://dhis2.example.org, https://app.example.org,")

        assert get_cors_origins() == ["https://dhis2.example.org", "https://app.example.org"]

    def test_strips_a_trailing_slash(self, monkeypatch):
        monkeypatch.setenv(CORS_ORIGINS_ENV_VAR, "https://dhis2.example.org/")

        assert get_cors_origins() == ["https://dhis2.example.org"]
