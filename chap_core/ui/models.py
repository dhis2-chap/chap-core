"""Model sources for the UI: the marketplace, your saved models, a checkout's models, and running
chapkit services, whether chap started them or a chaps deployment did."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from chap_core.ui.services import REPO_ROOT

if TYPE_CHECKING:
    from chap_core.services.model_marketplace import MarketplaceModel

# Label on containers chap started itself, so it knows it may stop them.
SERVICE_LABEL = "org.dhis2.chap.model"
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
    kind: str  # chapkit | saved | github | local
    source: str
    summary: str
    status: str
    tone: str  # good | warn | bad | neutral
    tags: tuple[str, ...]
    model_name: str | None = None  # value for --model-name, for models that do not need a service
    image: str | None = None  # docker image, for chapkit models
    repository: str | None = None
    service_id: str | None = None  # chapkit service id, as chaps and compose name it


@dataclass(frozen=True)
class SavedModel:
    """A model the user ran or added, remembered in the models file."""

    name: str
    model: str


@dataclass(frozen=True)
class ChapsModel:
    """A model in a chaps deployment, as `chaps status` reports it."""

    service_id: str
    state: str
    url: str | None

    @property
    def answering(self) -> bool:
        """Whether the model takes requests: chaps says registered, running or up."""
        return self.url is not None and self.state in ("registered", "running-not-registered", "unmanaged", "up")


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


def catalog_entries(marketplace: list[MarketplaceModel], saved: list[SavedModel] | None = None) -> list[CatalogEntry]:
    """Marketplace models, your saved models, and a checkout's GitHub models and examples as one list."""
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
                service_id=m.service_id,
            )
        )
    entries.extend(
        CatalogEntry(
            id=f"saved:{model.model}",
            name=model.name,
            kind="saved",
            source="Your models",
            summary=model.model,
            status="Saved",
            tone="neutral",
            tags=("Yours",),
            model_name=model.model,
        )
        for model in saved or []
    )
    for g in github_models():
        org = g.url.rstrip("/").split("/")[-2]
        entries.append(
            CatalogEntry(
                id=g.name,
                name=g.name,
                kind="github",
                source=f"GitHub · {org} · from the chap-core checkout",
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
            source="Example · from the chap-core checkout",
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
        name=f"chap-{model_id.replace('_', '-')}",
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


def models_file() -> Path:
    """Where your models are listed: CHAP_MODELS_FILE, else models.yaml in the runs folder."""
    from chap_core.ui.services import get_runs_dir

    path = os.environ.get("CHAP_MODELS_FILE")
    return Path(path).resolve() if path else get_runs_dir() / "models.yaml"


def saved_models(path: Path) -> list[SavedModel]:
    """The models in a models file; an absent or unreadable file lists none."""
    try:
        entries = yaml.safe_load(path.read_text()) or []
    except (OSError, yaml.YAMLError):
        return []
    return [
        SavedModel(str(e.get("name") or e["model"]), str(e["model"]))
        for e in entries
        if isinstance(e, dict) and e.get("model")
    ]


def remember_model(path: Path, model: str, name: str | None = None) -> None:
    """Add a model to the models file unless it is listed already.

    Addresses of chapkit services are not remembered: they change whenever a service restarts,
    and the marketplace already lists those models.
    """
    if model.startswith("http") and "github.com" not in model:
        return
    models = saved_models(path)
    if any(m.model == model for m in models):
        return
    models.append(SavedModel(name or model_label(model), model))
    _write_models(path, models)


def forget_model(path: Path, model: str) -> None:
    """Remove a model from the models file."""
    _write_models(path, [m for m in saved_models(path) if m.model != model])


def _write_models(path: Path, models: list[SavedModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump([{"name": m.name, "model": m.model} for m in models], sort_keys=False))


def chaps_binary() -> str | None:
    """The chaps program, when it is installed."""
    return shutil.which("chaps")


def chaps_project() -> Path | None:
    """The chaps deployment to use: CHAPS_PROJECT_DIR, the folder chap ui was started in when it is
    one, or the deployment chap ui created in the runs folder earlier."""
    from chap_core.ui.services import get_runs_dir

    configured = os.environ.get("CHAPS_PROJECT_DIR")
    candidates = [Path(configured)] if configured else [Path.cwd(), get_runs_dir() / "chaps"]
    return next((p.resolve() for p in candidates if (p / ".chaps").is_dir()), None)


def chaps_models(project: Path) -> dict[str, ChapsModel]:
    """The models of a chaps deployment by service id, with their state and address."""
    from chap_core.ui.services import run_external

    binary = chaps_binary()
    if binary is None:
        return {}
    result = run_external([binary, "--json", "-C", str(project), "status"], timeout=60)
    try:
        status = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {}
    return {
        m["id"]: ChapsModel(
            m["id"], m.get("state", ""), m.get("reach") if str(m.get("reach", "")).startswith("http") else None
        )
        for m in status.get("models", [])
    }


def chaps_start(model_id: str, project: Path | None) -> tuple[Path, str]:
    """Enable a marketplace model in a chaps deployment and start it.

    Without a deployment, a models-only one is created as chaps/ in the runs folder. Returns the
    deployment and what chaps printed.
    """
    from chap_core.ui.services import get_runs_dir, run_external

    binary = chaps_binary()
    if binary is None:
        raise RuntimeError("chaps is not installed")
    output = []
    if project is None:
        project = get_runs_dir() / "chaps"
        project.parent.mkdir(parents=True, exist_ok=True)
        output.append(_check(run_external([binary, "init", str(project), "--only", "none", "--models", "none"])))
    enabled = run_external([binary, "-C", str(project), "models", "enable", model_id, "--port", "auto"])
    if enabled.returncode != 0 and "already enabled" not in (enabled.stdout + enabled.stderr):
        _check(enabled)
    output.append(enabled.stdout + enabled.stderr)
    output.append(_check(run_external([binary, "-C", str(project), "up"])))
    return project, "\n".join(output)


def chaps_stop(model_id: str, project: Path) -> str:
    """Disable a model in a chaps deployment, which stops its container."""
    from chap_core.ui.services import run_external

    binary = chaps_binary()
    if binary is None:
        raise RuntimeError("chaps is not installed")
    return _check(run_external([binary, "-C", str(project), "models", "disable", model_id]))


def _check(result) -> str:
    output = (result.stdout or "") + (result.stderr or "")
    if result.returncode != 0:
        raise RuntimeError(
            output.strip().splitlines()[-1] if output.strip() else f"chaps exited with {result.returncode}"
        )
    return output
