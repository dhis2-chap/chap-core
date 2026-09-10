from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import yaml


@pytest.fixture
def marketplace_model():
    # Registry schema v2 example from dhis2-chap/model-marketplace.
    path = Path(__file__).parents[1] / "fixtures/marketplace/chapkit_simple_multistep_model.yaml"
    return yaml.safe_load(path.read_text())


@pytest.fixture
def marketplace_http(monkeypatch, marketplace_model):
    requests = []

    def respond(request):
        requests.append(str(request.url))
        if request.url.path.endswith("/registry.yaml"):
            content = {"schema_version": 2, "models": [f"models/{marketplace_model['id']}.yaml"]}
        else:
            content = marketplace_model
        return httpx.Response(200, text=yaml.safe_dump(content))

    client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: client(transport=httpx.MockTransport(respond), **kwargs))
    return requests


@pytest.fixture
def model_deployment(tmp_path, monkeypatch, mocker):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    compose = tmp_path / "compose.yml"
    compose.write_text("services:\n  chap:\n    image: chap:test\n")
    deployments = []

    def run(command, **kwargs):
        overlay = [Path(command[index + 1]) for index, value in enumerate(command) if value == "-f"][-1]
        deployments.append(yaml.safe_load(overlay.read_text()))
        return SimpleNamespace(stdout="127.0.0.1:54321\n")

    runner = mocker.patch("chap_core.cli_endpoints.marketplace.subprocess.run", side_effect=run)
    return SimpleNamespace(
        overlay=tmp_path / "compose.marketplace.yml",
        local_overlay=tmp_path / ".chap/compose.models.yml",
        runner=runner,
        deployments=deployments,
    )
