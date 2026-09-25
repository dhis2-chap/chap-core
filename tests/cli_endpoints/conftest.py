import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
import yaml


@pytest.fixture
def marketplace_model():
    # Registry schema v2 example from dhis2-chap/model-marketplace.
    path = Path(__file__).parents[1] / "fixtures/marketplace/chapkit_simple_multistep_model.yaml"
    return yaml.safe_load(path.read_text())


CUSTOM_SERVICE_INFO = {
    "id": "custom-model",
    "display_name": "Custom Model",
    "version": "1.0.0",
    "git_revision": "c" * 40,
    "model_metadata": {"author": "Someone"},
    "period_type": "monthly",
}


class FakeChap:
    """In-memory stand-in for the CHAP endpoints chap-admin drives, served over httpx.MockTransport."""

    def __init__(self):
        self.templates: list[dict] = []
        self.configured_models: list[dict] = []
        # A custom service that has self-registered from the compose service name chap-admin gives it.
        self.services: list[dict[str, Any]] = [
            {"id": "custom-model", "url": "http://marketplace-custom:8000", "info": CUSTOM_SERVICE_INFO}
        ]
        self.requests: list[tuple[str, str, object]] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        path: str = request.url.path
        body: dict = json.loads(request.content) if request.content else {}
        self.requests.append((request.method, path, body or None))
        if (request.method, path) == ("GET", "/v1/crud/model-templates"):
            return httpx.Response(200, json=[t for t in self.templates if t["isLive"]])
        if (request.method, path) == ("POST", "/v1/crud/model-templates"):
            return self._store_template(body)
        if (request.method, path) == ("POST", "/v1/crud/model-templates/from-service"):
            service = next((s for s in self.services if s["id"] == body["service_id"]), None)
            if service is None:
                return httpx.Response(404, json={"detail": "Service not found"})
            info = service["info"]
            return self._store_template(
                {"name": info["id"], "version": info["version"], "source_digest": info["git_revision"]}
            )
        if request.method == "DELETE" and path.startswith("/v1/crud/model-templates/"):
            template = next(t for t in self.templates if t["id"] == int(path.rsplit("/", 1)[1]))
            template["archived"] = True
            for model in self.configured_models:
                if model["modelTemplateId"] == template["id"]:
                    model["archived"] = True
            return httpx.Response(200, json={"message": "deleted"})
        if (request.method, path) == ("POST", "/v1/crud/configured-models"):
            model = {"id": len(self.configured_models) + 1, "archived": False, **body}
            model["modelTemplateId"] = model.pop("model_template_id")
            # Like CHAP, an identical configuration is returned, not stored twice.
            identity = {key: model[key] for key in ("modelTemplateId", "name", "user_option_values")}
            stored = next((m for m in self.configured_models if {k: m[k] for k in identity} == identity), None)
            if stored is not None:
                stored["archived"] = False
                return httpx.Response(200, json=stored)
            self.configured_models.append(model)
            return httpx.Response(200, json=model)
        if (request.method, path) == ("GET", "/v2/services"):
            return httpx.Response(200, json={"count": len(self.services), "services": self.services})
        return httpx.Response(404, json={"detail": f"No fake route for {request.method} {path}"})

    def _store_template(self, body: dict) -> httpx.Response:
        stored = next((t for t in self.templates if (t["name"], t["version"]) == (body["name"], body["version"])), None)
        if stored is not None:
            if stored["sourceDigest"] is not None and stored["sourceDigest"] != body.get("source_digest"):
                return httpx.Response(409, json={"detail": "A version is write-once"})
            stored["archived"] = False
            return httpx.Response(200, json=stored)
        for template in self.templates:
            if template["name"] == body["name"]:
                template["isLive"] = False
        template = {
            "id": len(self.templates) + 1,
            "name": body["name"],
            "version": body["version"],
            "sourceDigest": body.get("source_digest"),
            "sourceUrl": body.get("source_url"),
            "usesChapkit": body.get("uses_chapkit", False),
            "isLive": True,
            "archived": False,
        }
        self.templates.append(template)
        return httpx.Response(200, json=template)


@pytest.fixture
def fake_chap(monkeypatch, marketplace_model):
    """Serve the marketplace registry and a fake CHAP through every httpx.Client the commands build."""
    monkeypatch.delenv("CHAP_MARKETPLACE_URL", raising=False)
    monkeypatch.delenv("CHAP_URL", raising=False)
    monkeypatch.delenv("CHAP_API_TOKEN", raising=False)
    chap = FakeChap()
    chap.registry_requests = []

    def respond(request):
        if request.url.host in ("localhost", "chap.example.org"):
            return chap.handle(request)
        if request.url.host not in ("raw.githubusercontent.com", "models.example.org"):
            raise httpx.ConnectError(f"{request.url.host} is unreachable")
        chap.registry_requests.append(str(request.url))
        if request.url.path.endswith("/registry.yaml"):
            content = {"schema_version": 2, "models": [f"models/{marketplace_model['id']}.yaml"]}
        else:
            content = marketplace_model
        return httpx.Response(200, text=yaml.safe_dump(content))

    client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: client(transport=httpx.MockTransport(respond), **kwargs))
    return chap


@pytest.fixture
def marketplace_http(fake_chap):
    """The registry URLs fetched so far."""
    return fake_chap.registry_requests


@pytest.fixture
def model_deployment(tmp_path, monkeypatch, mocker, fake_chap):
    monkeypatch.chdir(tmp_path)
    compose = tmp_path / "compose.yml"
    compose.write_text("services:\n  chap:\n    image: chap:test\n")
    deployments = []

    def run(command, **kwargs):
        overlays = [Path(command[index + 1]) for index, value in enumerate(command) if value == "-f"]
        if overlays:
            deployments.append(yaml.safe_load(overlays[-1].read_text()))
        if "config" in command:
            stdout = '{"name": "chap-test"}'
        elif "{{.Config.WorkingDir}}" in command:
            stdout = "/app\n"
        elif "inspect" in command:
            stdout = "sha256:previous\n"
        else:
            stdout = ""
        return SimpleNamespace(stdout=stdout, returncode=0)

    runner = mocker.patch("chap_core.cli_endpoints.marketplace.subprocess.run", side_effect=run)
    return SimpleNamespace(
        overlay=tmp_path / "compose.marketplace.yml",
        runner=runner,
        deployments=deployments,
        chap=fake_chap,
    )
