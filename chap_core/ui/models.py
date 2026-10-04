"""Model sources for the UI: the marketplace, your saved models, a checkout's models, and running
chapkit services, whether chap started them or a chaps deployment did."""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from chap_core.ui.services import REPO_ROOT

if TYPE_CHECKING:
    from chap_core.services.model_marketplace import MarketplaceModel

# Label on containers chap started itself, so it knows it may stop them.
SERVICE_LABEL = "org.dhis2.chap.model"
CHAPKIT_PORT = "8000/tcp"
# The platform every chapkit image is built for, run under emulation where it is not native.
AMD64 = "linux/amd64"
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


# The `chaps ps` states of a model that is running and takes requests.
CHAPS_ANSWERING = frozenset({"up", "registered", "running, not registered", "unmanaged"})


@dataclass(frozen=True)
class ChapsModel:
    """A model chaps runs, as `chaps ps` reports it."""

    id: str
    service_id: str
    state: str
    url: str | None  # its own host address; None behind chap-core, where only a read-only proxy reaches it
    project_dir: Path  # the deployment or `chaps run` group it runs in, for logs, test and expose
    group: str | None = None  # the `chaps run` group; None in a deployment

    @property
    def internal(self) -> bool:
        """Running, but reachable only through chap-core's proxy, which forwards reads only.

        chap eval needs to write to the model, so such a model needs a host port of its own first.
        """
        return self.url is None and self.state in CHAPS_ANSWERING

    @property
    def answering(self) -> bool:
        return self.url is not None and self.state in CHAPS_ANSWERING


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
    name = f"chap-{model_id.replace('_', '-')}"
    # A container this UI started earlier that has since exited holds the name; it is replaced.
    for old in client.containers.list(all=True, filters={"name": f"^{name}$", "label": SERVICE_LABEL}):
        if old.status != "running":
            old.remove()
    options: dict[str, Any] = {
        "detach": True,
        "ports": {CHAPKIT_PORT: ("127.0.0.1", None)},
        "labels": {SERVICE_LABEL: model_id},
        "name": name,
    }
    try:
        container = client.containers.run(image, **options)
    except docker.errors.APIError as e:
        if "no matching manifest" not in str(e):
            raise
        # An image built for amd64 only, on an arm64 machine: Docker runs it under emulation.
        client.images.pull(image, platform=AMD64)
        container = client.containers.run(image, platform=AMD64, **options)
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


# Terminal colour and style codes, which models write to their logs and a log panel shows as text.
ANSI_CODES = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def plain_text(text: str) -> str:
    """Text without terminal colour codes."""
    return ANSI_CODES.sub("", text)


# Requests a model answers all the time: Docker's health check, and chap ui asking whether it answers.
ROUTINE_REQUESTS = ("/health", "/api/v1/info")


def newest_first(logs: str) -> str:
    """Log lines with the newest on top, leaving out the routine health and status requests."""
    lines = [line for line in logs.splitlines() if not any(path in line for path in ROUTINE_REQUESTS)]
    return "\n".join(reversed(lines))


def service_logs(service_id: str, tail: int = 200) -> str:
    import docker

    logs: bytes = docker.from_env().containers.get(service_id).logs(tail=tail)
    return plain_text(logs.decode(errors="replace"))


def service_info(url: str) -> dict | None:
    """The service's self-description, or None while it is not answering yet."""
    from chap_core.models.chapkit_rest_api_wrapper import CHAPKitRestAPIWrapper

    try:
        with CHAPKitRestAPIWrapper(url, timeout=2) as client:
            info: dict = client.info().model_dump(mode="json")
            return info
    except Exception:
        return None


@dataclass(frozen=True)
class ServiceMetrics:
    """What a chapkit service built with monitoring reports about itself. A field is None when the
    service does not report it, such as process figures outside Linux."""

    trainings: float | None
    predictions: float | None
    requests: float | None
    memory_bytes: float | None
    cpu_seconds: float | None
    started: float | None  # Unix time


# One sample of the Prometheus text format: a name, optional labels, and a value.
PROMETHEUS_SAMPLE = re.compile(r"^([a-zA-Z_:][\w:]*)(?:\{[^}]*\})?\s+(\S+)")


def parse_metrics(text: str) -> ServiceMetrics:
    """The figures chap ui shows, from a service's /metrics; samples of one name are summed over their labels."""
    totals: dict[str, float] = {}
    for line in text.splitlines():
        if (sample := PROMETHEUS_SAMPLE.match(line)) is not None:
            with contextlib.suppress(ValueError):
                totals[sample[1]] = totals.get(sample[1], 0.0) + float(sample[2])
    return ServiceMetrics(
        trainings=totals.get("ml_train_jobs_total"),
        predictions=totals.get("ml_predict_jobs_total"),
        # servicekit 3 names the request total after the stable HTTP semantic conventions; older services use
        # the experimental name.
        requests=totals.get(
            "http_server_request_duration_seconds_count", totals.get("http_server_duration_milliseconds_count")
        ),
        memory_bytes=totals.get("process_resident_memory_bytes"),
        cpu_seconds=totals.get("process_cpu_seconds_total"),
        started=totals.get("process_start_time_seconds"),
    )


def service_metrics(url: str) -> ServiceMetrics | None:
    """The service's metrics, or None when it has no /metrics: it was built without chapkit's monitoring."""
    import urllib.request

    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/metrics", timeout=2) as response:
            return parse_metrics(response.read().decode())
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


def chaps_registry_args() -> list[str]:
    """The marketplace chap ui lists models from, as chaps expects it: the URL of its registry.yaml.

    Empty when the default marketplace is used, so chaps keeps whatever its deployment recorded.
    """
    base = os.environ.get("CHAP_MARKETPLACE_URL")
    return ["--registry-url", f"{base.rstrip('/')}/registry.yaml"] if base else []


def chaps_binary() -> str | None:
    """The chaps program, when it is installed."""
    return shutil.which("chaps")


def _chaps() -> str:
    binary = chaps_binary()
    if binary is None:
        raise RuntimeError("chaps is not installed")
    return binary


def chaps_project() -> Path | None:
    """The chaps deployment to start models in: CHAPS_PROJECT_DIR, or the one the folder chap ui was
    started in belongs to, found the way chaps finds it. Without one, models start in chaps' default
    `chaps run` group."""
    configured = os.environ.get("CHAPS_PROJECT_DIR")
    start = (Path(configured) if configured else Path.cwd()).resolve()
    # The marker chaps itself looks for, so a stray `.chaps` folder is not taken for a deployment.
    return next((folder for folder in (start, *start.parents) if (folder / ".chaps" / "project.yaml").is_file()), None)


def chaps_group(project: Path | None) -> str | None:
    """The `chaps run` group chap ui starts models in: none in a deployment, else chaps' default group."""
    return None if project else "default"


def chaps_models(project: Path | None) -> list[ChapsModel]:
    """The models chaps runs: those of the deployment, or of every `chaps run` group."""
    if chaps_binary() is None:
        return []
    try:
        listed = _chaps_json(["ps"], project, timeout=60)
    except ChapsError:
        return []
    return [
        ChapsModel(
            m["id"],
            m["service_id"],
            m.get("state", ""),
            # Without a host port chaps reports chap-core's proxy URL, which chap eval cannot write through.
            m.get("url") if m.get("port") is not None else None,
            Path(m["project_dir"]),
            m.get("group"),
        )
        for m in listed.get("models", [])
    ]


def chaps_start(
    source: str,
    project: Path | None,
    model_id: str | None = None,
    port: int | None = None,
    everywhere: bool = False,
) -> dict:
    """Start a model with `chaps run`: a marketplace id, a repository URL or an image.

    It gets a free port unless `port` names one, and answers on this machine only unless
    `everywhere` publishes it on every address. Starting one that was started before reuses it.
    A template starts too: the user picked it, and without chaps it starts as any other image.
    Returns what chaps reports, with the model's URL.
    """
    args = ["run", "--no-wait", "--allow-template"]
    args += ["--id", model_id] if model_id else []
    args += ["--port", str(port)] if port else []
    args += ["--bind", "0.0.0.0"] if everywhere else []
    # The source is typed by the user: after `--`, one starting with `-` cannot pass for an option.
    return dict(_chaps_json([*args, "--", source], project))


def chaps_stop(model: ChapsModel, delete_data: bool = False) -> dict:
    """Stop a model. Its data stays, so starting it again picks up its configurations and trained
    models, unless `delete_data`. An added model's definition stays, so starting it again needs no
    download."""
    return dict(_chaps_json(["stop", model.id, *(["--purge"] if delete_data else [])], model.project_dir))


def chaps_expose(model: ChapsModel) -> dict:
    """Give a model behind chap-core a host port of its own, so chap eval can reach it."""
    _chaps_json(["models", "expose", model.id, "--port", "auto"], model.project_dir)
    return dict(_chaps_json(["up", "--wait"], model.project_dir))


def chaps_added_models(models: list[ChapsModel]) -> list[CatalogEntry]:
    """Running models that were started from a URL or an image rather than the marketplace."""
    entries = []
    for project_dir in sorted({m.project_dir for m in models}):
        try:
            listed = _chaps_json(["models", "list"], project_dir, timeout=60)
        except ChapsError:
            continue
        status, tone = ASSESSMENT["gray"]
        entries += [
            CatalogEntry(
                id=m["id"],
                name=m.get("display_name") or m["id"],
                kind="chapkit",
                source="Chapkit · started from a URL or image",
                summary=f"Started with `chaps run` from `{m['image']}`.",
                status=status,
                tone=tone,
                tags=("Added",),
                image=m["image"],
                service_id=m["service_id"],
            )
            for m in listed
            if m.get("manual") and m.get("enabled")
        ]
    return entries


def chaps_test(model: ChapsModel) -> str:
    """Have chaps train and predict with a running model on generated data."""
    from chap_core.ui.services import run_external

    return _check(run_external([_chaps(), "-C", str(model.project_dir), "models", "test", model.id]))


def chaps_logs(model: ChapsModel, tail: int = 200) -> str:
    """The last lines a model logged."""
    from chap_core.ui.services import run_external

    command = [_chaps(), "-C", str(model.project_dir), "logs", "--tail", str(tail), model.service_id]
    return str(run_external(command, timeout=60).stdout)


# GitHub's answer for a repository that does not exist, or that the caller may not see.
GITHUB_NOT_FOUND = re.compile(r"HTTP 404 from https://api\.github\.com/repos/([^/\s]+/[^/\s]+)")


def readable_error(message: str) -> str:
    """A chaps error in words a user can act on, where chaps passes on a raw HTTP answer."""
    if found := GITHUB_NOT_FOUND.search(message):
        return f"GitHub has no repository {found.group(1)}, or it is private: check the URL"
    return message


class ChapsError(RuntimeError):
    """A chaps command failed; the message is its error, `output` everything it printed."""

    def __init__(self, message: str, output: str):
        super().__init__(readable_error(message))
        self.output = output


def _chaps_json(args: list[str], project: Path | None, timeout: float = 1800):
    """Run a chaps command with --json and return its document; a failure raises ChapsError."""
    from chap_core.ui.services import run_external

    location = ["-C", str(project)] if project else []
    try:
        result = run_external([_chaps(), "--json", *chaps_registry_args(), *location, *args], timeout=timeout)
    except subprocess.TimeoutExpired:
        raise ChapsError(f"chaps did not answer within {timeout:.0f} seconds; is Docker running?", "") from None
    output = (result.stdout or "") + (result.stderr or "")
    try:
        document = json.loads(result.stdout)
    except json.JSONDecodeError:
        lines = output.strip().splitlines()
        raise ChapsError(lines[-1] if lines else f"chaps exited with {result.returncode}", output) from None
    if isinstance(document, dict) and document.get("ok") is False:
        # chaps gives the way out separately, in `hint`; the UI shows the whole sentence.
        message = "; ".join(part for part in (document.get("error"), document.get("hint")) if part)
        raise ChapsError(message or f"chaps exited with {result.returncode}", output)
    return document


def _check(result) -> str:
    output = (result.stdout or "") + (result.stderr or "")
    if result.returncode != 0:
        lines = output.strip().splitlines()
        errors = [line.removeprefix("error: ") for line in lines if line.startswith("error: ")]
        message = errors[0] if errors else lines[-1] if lines else f"chaps exited with {result.returncode}"
        raise ChapsError(message, output)
    return output
