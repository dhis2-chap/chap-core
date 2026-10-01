"""Model sources for the UI: marketplace entries, GitHub models and local chapkit containers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import yaml

from chap_core.ui.services import REPO_ROOT

if TYPE_CHECKING:
    from chap_core.services.model_marketplace import MarketplaceModel

SERVICE_LABEL = "chap-ui.model"
CHAPKIT_PORT = "8000/tcp"
CONFIGURED_MODELS_DIR = REPO_ROOT / "config" / "configured_models"


@dataclass(frozen=True)
class GithubModel:
    name: str
    url: str
    commit: str | None

    @property
    def model_name(self) -> str:
        """Value for --model-name, pinned to the stable commit when one is known."""
        return f"{self.url}@{self.commit}" if self.commit else self.url


@dataclass(frozen=True)
class ChapkitService:
    id: str
    name: str
    model_id: str
    image: str
    status: str
    url: str | None


def model_label(model_name: str) -> str:
    """Short name for a model: the chapkit model id for a local service, otherwise the last path part."""
    if model_name.startswith("http") and "github.com" not in model_name:
        try:
            match = next((s.model_id for s in list_services() if s.url and model_name.startswith(s.url)), None)
        except Exception:
            match = None
        if match:
            return match
    return model_name.split("@")[0].rstrip("/").split("/")[-1]


def marketplace_image(entry: MarketplaceModel) -> str:
    """Image for the stable version of an entry, or the latest when there is no stable channel."""
    wanted = entry.channels.get("stable") or entry.channels.get("latest")
    version = next((v for v in entry.versions if v.version == wanted), entry.versions[-1])
    return f"{entry.source.image}:{version.image_tag}"


def github_models() -> list[GithubModel]:
    """Models from the configured-models seed files, pinned to their stable commit."""
    models = []
    for path in sorted(CONFIGURED_MODELS_DIR.glob("*.yaml")):
        for entry in yaml.safe_load(path.read_text()) or []:
            url = entry["url"]
            versions = entry.get("versions") or {}
            commit = versions.get("stable") or (list(versions.values())[-1] if versions else None)
            models.append(
                GithubModel(entry.get("name") or url.rstrip("/").split("/")[-1], url, commit and commit.lstrip("@"))
            )
    return models


def start_service(image: str, model_id: str) -> ChapkitService:
    """Start a chapkit model image as a local container on a free port."""
    import docker

    client = docker.from_env()
    container = client.containers.run(
        image,
        detach=True,
        ports={CHAPKIT_PORT: ("127.0.0.1", None)},
        labels={SERVICE_LABEL: model_id},
        name=f"chap-ui-{model_id.replace('_', '-')}",
    )
    return _service(container)


def list_services() -> list[ChapkitService]:
    """Chapkit containers started from the UI."""
    import docker

    client = docker.from_env()
    return [_service(c) for c in client.containers.list(all=True, filters={"label": SERVICE_LABEL})]


def stop_service(service_id: str) -> None:
    """Stop and remove a chapkit container started from the UI."""
    import docker

    container = docker.from_env().containers.get(service_id)
    container.remove(force=True)


def service_logs(service_id: str, tail: int = 200) -> str:
    import docker

    logs: bytes = docker.from_env().containers.get(service_id).logs(tail=tail)
    return logs.decode(errors="replace")


def service_info(url: str) -> dict | None:
    """The service's self-description, or None while it is not answering yet."""
    from chap_core.models.chapkit_rest_api_wrapper import CHAPKitRestAPIWrapper

    try:
        with CHAPKitRestAPIWrapper(url, timeout=2) as client:
            info: dict = client.info().model_dump(mode="json")
            return info
    except Exception:
        return None


def _service(container) -> ChapkitService:
    container.reload()
    bindings = (container.ports or {}).get(CHAPKIT_PORT) or []
    url = f"http://localhost:{bindings[0]['HostPort']}" if bindings else None
    image = container.image.tags[0] if container.image.tags else container.image.short_id
    return ChapkitService(container.id, container.name, container.labels[SERVICE_LABEL], image, container.status, url)
