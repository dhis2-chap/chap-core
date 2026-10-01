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


# The model author's own maturity rating, as documented in docs/external_models/model_metadata.md.
ASSESSMENT = {
    "green": ("Validated", "good"),
    "yellow": ("Ready for testing", "good"),
    "orange": ("Limited data", "warn"),
    "red": ("Experimental", "bad"),
    "gray": ("Not for use", "neutral"),
}


@dataclass(frozen=True)
class CatalogEntry:
    """One model in the catalog, whichever source it comes from."""

    id: str
    name: str
    kind: str  # chapkit | github | local
    source: str
    summary: str
    status: str
    tone: str  # good | warn | bad | neutral
    tags: tuple[str, ...]
    model_name: str | None = None  # value for --model-name, for models that do not need a service
    image: str | None = None  # docker image, for chapkit models
    repository: str | None = None


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
    managed: bool  # started from the UI, so the UI may stop it; others belong to e.g. a chaps deployment


def catalog_entries(marketplace: list[MarketplaceModel]) -> list[CatalogEntry]:
    """Marketplace models, GitHub models and bundled examples as one list."""
    from chap_core.ui.services import example_models

    entries = []
    for m in marketplace:
        status, tone = ASSESSMENT.get(m.assessed_status or "gray", ASSESSMENT["gray"])
        compat, covariates = m.compatibility, m.covariates
        tags = [p.capitalize() for p in compat.period_types or []]
        tags += [c.replace("_", " ").capitalize() for c in covariates.required or []] or ["No covariates"]
        if compat.max_prediction_periods is not None:
            tags.append(f"Horizon {compat.min_prediction_periods or 0}-{compat.max_prediction_periods}")
        if m.kind == "template":
            tags.append("Template")
        version = m.channels.get("stable") or m.channels.get("latest") or ""
        entries.append(
            CatalogEntry(
                id=m.id,
                name=m.display_name or m.id,
                kind="chapkit",
                source=f"Chapkit · v{version}" if version else "Chapkit",
                summary=m.summary or "",
                status=status,
                tone=tone,
                tags=tuple(tags),
                image=marketplace_image(m),
                repository=m.source.repository,
            )
        )
    for g in github_models():
        org = g.url.rstrip("/").split("/")[-2]
        entries.append(
            CatalogEntry(
                id=g.name,
                name=g.name,
                kind="github",
                source=f"GitHub · {org}",
                summary="One of the models CHAP installs by default, run from its GitHub repository.",
                status="Installed by default",
                tone="good",
                tags=(*(p.capitalize() for p in ("monthly", "weekly") if p in g.name), "GitHub"),
                model_name=g.model_name,
                repository=g.url,
            )
        )
    entries.extend(
        CatalogEntry(
            id=path.name,
            name=path.name,
            kind="local",
            source="Local example",
            summary="Example model bundled with chap-core, useful for trying things out quickly.",
            status="Example",
            tone="neutral",
            tags=("Local",),
            model_name=str(path),
        )
        for path in example_models()
    )
    return entries


def model_kind(model_name: str) -> str:
    """Where a model comes from, in words."""
    if "github.com" in model_name:
        return "GitHub repository"
    if model_name.startswith("http"):
        return "Chapkit service"
    return "Local folder"


def model_label(model_name: str) -> str:
    """Short name for a model: a chapkit service's own display name, otherwise the last path part."""
    if model_name.startswith("http") and "github.com" not in model_name:
        name = service_name(model_name)
        if name:
            return name
    return model_name.split("@")[0].rstrip("/").split("/")[-1]


_service_names: dict[str, str] = {}


def service_name(url: str) -> str | None:
    """Display name a chapkit service reports about itself, remembered once it has answered."""
    url = url.rstrip("/")
    if url not in _service_names:
        info = service_info(url)
        if not info:
            return None
        _service_names[url] = info.get("display_name") or info.get("id") or url
    return _service_names[url]


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
    container.reload()
    return _service(container, model_id)


def list_services(images: dict[str, str] | None = None) -> list[ChapkitService]:
    """Chapkit containers started from the UI, plus running containers of known model images.

    `images` maps an image repository (without tag) to its catalog id, so models deployed by
    other tools, such as chaps, are recognised too.
    """
    import docker

    services = []
    for container in docker.from_env().containers.list(all=True):
        model_id = container.labels.get(SERVICE_LABEL)
        if model_id is None and images and container.status == "running":
            repository = (container.image.tags[0] if container.image.tags else "").rsplit(":", 1)[0]
            model_id = images.get(repository)
        if model_id is not None:
            services.append(_service(container, model_id))
    return services


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


def _service(container, model_id: str) -> ChapkitService:
    bindings = (container.ports or {}).get(CHAPKIT_PORT) or []
    url = f"http://localhost:{bindings[0]['HostPort']}" if bindings else None
    image = container.image.tags[0] if container.image.tags else container.image.short_id
    managed = SERVICE_LABEL in container.labels
    return ChapkitService(container.id, container.name, model_id, image, container.status, url, managed)
