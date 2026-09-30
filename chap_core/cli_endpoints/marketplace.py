"""Install and update chapkit model services in a CHAP Compose deployment.

``install``, ``install-all``, ``update`` and ``uninstall`` are the ``chap-admin`` commands.
They edit the deployment's Compose overlay, start the model service, and register it in
the running CHAP instance over its REST API once it is up.
"""

import json
import logging
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import httpx
import yaml
from cyclopts import Parameter

if TYPE_CHECKING:
    from chap_core.services.chap_api import ChapApi
    from chap_core.services.model_marketplace import ModelPin

logger = logging.getLogger(__name__)

ModelArg = Annotated[str, Parameter(help="Marketplace model ID, or a name for a custom chapkit model.")]
ComposeArg = Annotated[
    tuple[Path, ...],
    Parameter(help="Base Compose files in deployment order. Defaults to compose.yml."),
]
ImageArg = Annotated[str | None, Parameter(help="Custom chapkit container image (requires --accept-risk).")]
RiskArg = Annotated[
    bool,
    Parameter(help="Accept responsibility for running unreviewed custom model code and using its forecasts."),
]
PlatformArg = Annotated[str | None, Parameter(help="Container platform, e.g. linux/amd64 for R-INLA on Apple Silicon.")]
DeleteDataArg = Annotated[bool, Parameter(help="Also delete the model's data volume. This cannot be undone.")]
UrlArg = Annotated[str | None, Parameter(help="CHAP URL. Defaults to CHAP_URL or http://localhost:8000.")]
TokenArg = Annotated[str | None, Parameter(help="CHAP API token. Defaults to CHAP_API_TOKEN.")]

# How long a freshly started service gets to self-register with CHAP.
REGISTRATION_TIMEOUT = 60

# What a command reports as a failure instead of a traceback.
DEPLOYMENT_ERRORS = (ValueError, OSError, httpx.HTTPError, yaml.YAMLError, subprocess.CalledProcessError)


def install(
    model: ModelArg,
    *,
    compose_file: ComposeArg = (Path("compose.yml"),),
    image: ImageArg = None,
    accept_risk: RiskArg = False,
    platform: PlatformArg = None,
    url: UrlArg = None,
    token: TokenArg = None,
) -> None:
    """Install a marketplace model's verified stable version into a running CHAP deployment.

    Run from your CHAP Docker Compose deployment directory. The service is started, CHAP
    stores its model template from the running service, and the model's verified
    configurations are added from the marketplace entry. A service that does not come up
    and register is removed again. Custom chapkit images can be installed with --image
    IMAGE --accept-risk; they get one default configuration. Docker Compose is required.
    """
    from chap_core.services.chap_api import ChapApi

    with ChapApi(url, token) as api:
        _exit_on_error(lambda: _deploy(api, model, compose_file, image, accept_risk, platform, updating=False))


def install_all(
    *,
    compose_file: ComposeArg = (Path("compose.yml"),),
    accept_risk: RiskArg = False,
    platform: PlatformArg = None,
    url: UrlArg = None,
    token: TokenArg = None,
) -> None:
    """Install every marketplace model that has a verified stable version.

    Runs the same installation as ``install`` for each model the marketplace lists,
    skipping models that are already installed, templates for model authors, and models
    without a verified stable version. Fails at the end if any model could not be
    installed; the others stay installed.
    """
    from chap_core.services.chap_api import ChapApi
    from chap_core.services.model_marketplace import list_models, registry_url, stable_pin

    def run() -> None:
        with ChapApi(url, token) as api:
            _check_chap(api)
            entries, invalid = list_models(registry_url())
            for model_file, reason in invalid.items():
                logger.error("Could not read %s from the marketplace: %s", model_file, reason)
            failed = list(invalid)
            installed = _load_overlay(compose_file)[1]["services"]
            for entry in entries:
                try:
                    stable_pin(entry)
                except ValueError as reason:
                    logger.info("Skipping %s: %s", entry.id, reason)
                    continue
                if _service_name(entry.id) in installed:
                    logger.info("Skipping %s: already installed.", entry.id)
                    continue
                try:
                    # Validates the model name, so a badly named entry fails on its own.
                    _deploy(api, entry.id, compose_file, None, accept_risk, platform, updating=False)
                except DEPLOYMENT_ERRORS as error:
                    logger.error("Could not install %s: %s", entry.id, error)
                    failed.append(entry.id)
            if failed:
                raise ValueError(f"Could not install {', '.join(failed)}.")

    _exit_on_error(run)


def update(
    model: ModelArg,
    *,
    compose_file: ComposeArg = (Path("compose.yml"),),
    image: ImageArg = None,
    accept_risk: RiskArg = False,
    platform: PlatformArg = None,
    url: UrlArg = None,
    token: TokenArg = None,
) -> None:
    """Update an installed model to the marketplace's verified stable version.

    The new version is registered in CHAP as a new template version with its
    configurations once the new service is up; earlier versions and their backtests are
    untouched. Custom models require --accept-risk again. Use --image to select a new
    custom image, or omit it to pull the previously installed custom image reference.
    """
    from chap_core.services.chap_api import ChapApi

    with ChapApi(url, token) as api:
        _exit_on_error(lambda: _deploy(api, model, compose_file, image, accept_risk, platform, updating=True))


def uninstall(
    model: ModelArg,
    *,
    compose_file: ComposeArg = (Path("compose.yml"),),
    delete_data: DeleteDataArg = False,
    url: UrlArg = None,
    token: TokenArg = None,
) -> None:
    """Remove an installed model service from a CHAP deployment.

    The model template and its configured models are retired in CHAP, never deleted,
    since backtests reference them. The model's data volume is kept so a later install
    resumes from it; pass --delete-data to remove it permanently.
    """
    from chap_core.services.chap_api import ChapApi

    with ChapApi(url, token) as api:
        _exit_on_error(lambda: _remove(model, compose_file, delete_data=delete_data, api=api))


def _remove(model: str, compose_file: tuple[Path, ...], *, delete_data: bool, api: "ChapApi") -> None:
    overlay, config, service_name = _prepare(model, compose_file)
    previous = config["services"].pop(service_name, None)
    if previous is None:
        raise ValueError(f"Model '{model}' is not installed.")
    _retire_template(api, previous.get("x-chap-template"))
    volume = f"{service_name}-data"
    config["volumes"].pop(volume, None)
    command = [*_compose_command(compose_file), "-f", str(overlay)]
    project = None
    if delete_data:
        listing = subprocess.run(
            [*command, "config", "--format", "json"], check=True, stdout=subprocess.PIPE, text=True
        )
        project = json.loads(listing.stdout)["name"]
    # Remove the container while the overlay still declares it, then publish the pruned file.
    subprocess.run([*command, "rm", "--stop", "--force", service_name], check=True)
    pending = _write_pending(config, overlay)
    try:
        pending.replace(overlay)
    finally:
        pending.unlink(missing_ok=True)
    # Delete the volume last so a failure here cannot leave the service declared without a
    # container, and do not fail the uninstall over it: the model itself is already gone.
    if project is not None and subprocess.run(["docker", "volume", "rm", f"{project}_{volume}"]).returncode:
        logger.warning("Could not delete volume %s_%s; remove it with 'docker volume rm'.", project, volume)
    logger.info("Uninstalled %s.%s", model, "" if delete_data else f" Its data volume '{volume}' was kept.")


def _retire_template(api: "ChapApi", template_name: str | None) -> None:
    """Archive every version of the template the overlay recorded for the service, if CHAP still lists it."""
    if template_name is None:
        logger.info("The overlay records no model template for this service, so nothing is retired in CHAP.")
        return
    live = [template for template in api.runnable_model_templates() if template["name"] == template_name]
    if not live:
        logger.info("Model template %s is not listed by CHAP at %s, so nothing is retired.", template_name, api.url)
        return
    api.archive_model_template(live[0]["id"], all_versions=True)
    logger.info("Retired model template %s and its configured models in CHAP at %s.", template_name, api.url)


def _prepare(model: str, compose_files: tuple[Path, ...]) -> tuple[Path, dict[str, Any], str]:
    """Validate the arguments and return the overlay path, its config and the service name."""
    if not re.fullmatch(r"[a-z0-9][a-z0-9_]*", model):
        raise ValueError("Model names must contain only lowercase letters, digits and underscores.")
    overlay, config = _load_overlay(compose_files)
    return overlay, config, _service_name(model)


def _load_overlay(compose_files: tuple[Path, ...]) -> tuple[Path, dict[str, Any]]:
    """Return the deployment's model overlay path and its validated config."""
    if not compose_files or any(not path.is_file() for path in compose_files):
        raise ValueError("Run from a CHAP deployment directory or pass its base files with --compose-file.")
    overlay = compose_files[0].resolve().parent / "compose.marketplace.yml"
    config = yaml.safe_load(overlay.read_text()) if overlay.exists() else {"services": {}, "volumes": {}}
    if (
        not isinstance(config, dict)
        or not isinstance(config.get("services"), dict)
        or not all(isinstance(service, dict) for service in config["services"].values())
        or not isinstance(config.get("volumes", {}) or {}, dict)
    ):
        raise ValueError(f"Invalid model Compose file: {overlay}")
    config["volumes"] = config.get("volumes") or {}
    return overlay, config


def _service_name(model: str) -> str:
    return f"marketplace-{model.replace('_', '-')}"


def _compose_command(compose_files: tuple[Path, ...]) -> list[str]:
    command = ["docker", "compose"]
    for path in compose_files:
        command.extend(["-f", str(path.resolve())])
    return command


def _write_pending(config: dict[str, Any], overlay: Path) -> Path:
    """Write the new model Compose file beside the overlay it will replace."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", dir=overlay.parent, delete=False) as temporary:
        yaml.safe_dump(config, temporary, sort_keys=False)
        pending = Path(temporary.name)
    # NamedTemporaryFile creates mode 0600; the overlay must stay readable to other operators.
    pending.chmod(0o644)
    return pending


def _inspect_image(image: str, template: str) -> str | None:
    """Return a field of the local image, or None when the reference is not present locally."""
    result = subprocess.run(
        ["docker", "image", "inspect", "--format", template, image],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return result.stdout.strip() if not result.returncode else None


def _exit_on_error(action) -> None:
    """Run a command body, reporting a deployment error as a failed command."""
    from chap_core.log_config import initialize_logging

    initialize_logging()
    try:
        action()
    except DEPLOYMENT_ERRORS as error:
        logger.error("%s", error)
        raise SystemExit(1) from error


def _check_chap(api: "ChapApi") -> None:
    """Fail before Docker is touched when CHAP cannot be reached."""
    api.services()


def _register_service(
    api: "ChapApi", service_name: str, pin: "ModelPin | None", taken: set[str], previous_name: str | None
) -> str:
    """Store the template of a just started service in CHAP and give it its configurations.

    The service's own id is only known once it has registered, so it is found by the
    hostname it registered from; an update drops the previous container's registration
    before starting the new one, so the registration seen is the new container's. A
    marketplace service must report the entry's service id and the pinned commit. A
    custom service must not register under a model name in ``taken``, which another
    install already serves. An update must register under the ``previous_name`` it
    replaces, since another id is another model. Returns the template name.

    These checks keep chap-admin from mixing models, not a boundary of the registry: its
    REST API lets the latest registration of an id win, so a custom service that claims a
    taken id routes that model's runs to itself until it is refused and deregistered.
    """
    from chap_core.services.model_marketplace import configured_model_requests

    deadline = time.monotonic() + REGISTRATION_TIMEOUT
    while True:
        registered = [service for service in api.services() if _hostname(service) == service_name]
        for service in registered:
            if pin is not None and service["id"] != pin.entry.service_id:
                raise ValueError(
                    f"Service {service_name} registered as '{service['id']}', but the marketplace entry is "
                    f"'{pin.entry.service_id}'. Refusing to register it under another model's name."
                )
            if previous_name is not None and service["id"] != previous_name:
                raise ValueError(
                    f"Service {service_name} registered as '{service['id']}', but this installation serves "
                    f"'{previous_name}'. Another id is another model: uninstall this one and install the new image."
                )
            if pin is None and service["id"] in taken:
                raise ValueError(
                    f"Service {service_name} registered as '{service['id']}', which is already a model in "
                    f"CHAP at {api.url}. Refusing to replace it with a custom image."
                )
        if pin is not None:
            registered = [service for service in registered if service["info"].get("git_revision") == pin.commit]
        if registered:
            break
        if time.monotonic() > deadline:
            expected = "" if pin is None else f" with git revision {pin.commit}"
            raise ValueError(
                f"Service {service_name} did not register with CHAP at {api.url}{expected} within "
                f"{REGISTRATION_TIMEOUT} seconds. Images must support chapkit self-registration and "
                "report the commit they were built from."
            )
        time.sleep(2)
    template = api.create_model_template_from_service(registered[0]["id"])
    if pin is None:
        requests = [{"name": "default", "model_template_id": template["id"], "user_option_values": {}}]
    else:
        requests = configured_model_requests(pin, template["id"])
    for request in requests:
        api.create_configured_model(request)
    logger.info(
        "Registered model template %s version %s (id %s) with %d configured models in CHAP at %s.",
        template["name"],
        template["version"],
        template["id"],
        len(requests),
        api.url,
    )
    return str(template["name"])


def _hostname(service: dict[str, Any]) -> str:
    return str(service["url"]).split("//", 1)[-1].split(":")[0]


def _undo_registration(api: "ChapApi", service_name: str, pin: "ModelPin | None", live_before: set[int]) -> None:
    """Take back what a failed run left in CHAP, before its container is removed or rolled back.

    A template of this model that is live now but was not before the run was made live by
    it: by the service's own registration, which makes a first version live at once, or
    by the configurations posted for it. Retiring it hands live status back to the version
    the restored container runs. The run's registrations are dropped, so a service whose
    registration it overwrote registers again.
    """
    registered = [service for service in api.services() if _hostname(service) == service_name]
    names = {service["id"] for service in registered} | ({pin.entry.service_id} if pin is not None else set())
    for template in api.model_templates():
        if template["name"] in names and template["id"] not in live_before:
            api.archive_model_template(template["id"])
            logger.info("Retired model template %s version %s in CHAP.", template["name"], template["version"])
    for service in registered:
        api.deregister_service(service["id"])


def _deploy(
    api: "ChapApi",
    model: str,
    compose_files: tuple[Path, ...],
    image: str | None,
    accept_risk: bool,
    platform: str | None,
    updating: bool,
) -> None:
    from chap_core.services.model_marketplace import DEFAULT_REGISTRY_URL, registry_url, resolve_model

    overlay, config, service_name = _prepare(model, compose_files)
    services = config["services"]
    previous = services.get(service_name)
    if updating and previous is None:
        raise ValueError(f"Model '{model}' is not installed. Run 'chap-admin install {model}' first.")
    if not updating and previous is not None:
        raise ValueError(f"Model '{model}' is already installed. Run 'chap-admin update {model}' instead.")

    registry = (previous.get("x-chap-registry") if previous is not None else None) or registry_url()
    custom = image is not None or (previous is not None and previous.get("x-chap-custom", False))
    if custom or registry != DEFAULT_REGISTRY_URL:
        source = "Custom models" if custom else f"Models from '{registry}'"
        warning = (
            f"{source} are not reviewed by the CHAP marketplace. You accept responsibility "
            "for running their code, sharing data with them, and using their forecasts."
        )
        if not accept_risk:
            raise ValueError(f"{warning} Pass --accept-risk to continue.")
        logger.warning(warning)
    if custom:
        image = image if image is not None else previous["image"]
        if not image or any(character.isspace() for character in image) or "$" in image:
            raise ValueError("Provide a valid custom container image reference.")
        version = "custom"
        pin = None
    else:
        pin = resolve_model(model, registry)
        image, version = pin.image, pin.version
    _check_chap(api)
    live_templates = api.runnable_model_templates()
    live_before = {template["id"] for template in live_templates}
    own_template = previous.get("x-chap-template") if previous is not None else None
    taken = {template["name"] for template in live_templates} - {own_template}

    # Retain operator settings and the data volume when updating a model.
    if previous is not None:
        service = dict(previous)
    else:
        service = {
            "restart": "unless-stopped",
            "init": True,
            "read_only": True,
            "security_opt": ["no-new-privileges:true"],
            "cap_drop": ["ALL"],
            "environment": {
                "SERVICEKIT_ORCHESTRATOR_URL": "http://chap:8000/v2/services/$$register",
                "SERVICEKIT_REGISTRATION_KEY": "${SERVICEKIT_REGISTRATION_KEY:-}",
                "SERVICEKIT_HOST": service_name,
            },
            "depends_on": {"chap": {"condition": "service_healthy"}},
        }
        config["volumes"][f"{service_name}-data"] = {}
    service.update({"image": image, "x-chap-custom": bool(custom), "x-chap-version": version})
    if not custom:
        service["x-chap-registry"] = registry
    if platform is not None:
        if not re.fullmatch(r"[a-z0-9][a-z0-9/._-]*", platform):
            raise ValueError("Provide a valid container platform, for example linux/amd64.")
        service["platform"] = platform
    services[service_name] = service

    command = _compose_command(compose_files)
    previous_image_id = None
    if previous is not None and "@" not in previous["image"]:
        # A pull can move a tag but not a digest; keep the ID so a rollback can move the tag back.
        previous_image_id = _inspect_image(previous["image"], "{{.Id}}")
    pull = ["docker", "pull", *(["--platform", service["platform"]] if "platform" in service else []), image]
    try:
        subprocess.run(pull, check=True)
    except subprocess.CalledProcessError as error:
        # Docker has already printed why the pull failed, so add a hint instead of repeating it.
        hint = ""
        if "platform" not in service:
            hint = (
                " If this model publishes no image for your machine's architecture, retry with:"
                f" chap-admin {'update' if updating else 'install'} {model} --platform linux/amd64"
            )
        raise ValueError(f"Could not pull {image}.{hint}") from error
    if previous is None:
        # chapkit keeps its SQLite database under data/ in the image's working directory, which
        # the image owns for its service user; mounting elsewhere leaves a root-owned volume.
        workdir = (_inspect_image(image, "{{.Config.WorkingDir}}") or "/").rstrip("/")
        service["volumes"] = [f"{service_name}-data:{workdir}/data", {"type": "tmpfs", "target": "/tmp"}]
    # Publish the new pin only after the service runs and is registered in CHAP.
    pending = _write_pending(config, overlay)
    try:
        pending_command = [*command, "-f", str(pending)]
        up = ["up", "-d", "--no-deps", "--wait", "--wait-timeout", "120", service_name]
        try:
            if previous is not None and previous.get("x-chap-template"):
                api.deregister_service(previous["x-chap-template"])
            subprocess.run([*pending_command, *up], check=True)
            service["x-chap-template"] = _register_service(api, service_name, pin, taken, own_template)
        except (*DEPLOYMENT_ERRORS, KeyboardInterrupt):
            try:
                _undo_registration(api, service_name, pin, live_before)
            except DEPLOYMENT_ERRORS as error:
                logger.warning("Could not take back the registration in CHAP: %s", error)
            if previous is not None:
                logger.warning("Update failed; restoring the previous model service.")
                if previous_image_id is not None:
                    subprocess.run(["docker", "tag", previous_image_id, previous["image"]], check=True)
                subprocess.run([*command, "-f", str(overlay), *up], check=True)
            else:
                logger.warning("Installation failed; removing the model service.")
                subprocess.run([*pending_command, "rm", "--stop", "--force", service_name], check=True)
            raise
        pending.unlink()
        pending = _write_pending(config, overlay)
        pending.replace(overlay)
    finally:
        pending.unlink(missing_ok=True)
    logger.info("%s %s (%s): %s", "Updated" if updating else "Installed", model, version, image)
    logger.info("Include -f %s in future Docker Compose commands for this deployment.", overlay)
