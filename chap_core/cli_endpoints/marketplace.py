"""Install and update chapkit model services in a CHAP Compose deployment.

``install``, ``update`` and ``uninstall`` are the ``chap-admin`` commands. They edit the
deployment's Compose overlay and tell the running CHAP instance about the model over its
REST API. ``start`` and ``stop`` are the ``chap model`` commands for a standalone local
service used by ``chap eval``; they never talk to a CHAP server.
"""

import json
import logging
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

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
NoStartArg = Annotated[
    bool, Parameter(negative="", help="Register the model in CHAP and write the overlay without starting it.")
]
UrlArg = Annotated[str | None, Parameter(help="CHAP URL. Defaults to CHAP_URL or http://localhost:8000.")]
TokenArg = Annotated[str | None, Parameter(help="CHAP API token. Defaults to CHAP_API_TOKEN.")]

# How long a freshly started custom service gets to self-register with CHAP.
REGISTRATION_TIMEOUT = 60


def install(
    model: ModelArg,
    *,
    compose_file: ComposeArg = (Path("compose.yml"),),
    image: ImageArg = None,
    accept_risk: RiskArg = False,
    platform: PlatformArg = None,
    no_start: NoStartArg = False,
    url: UrlArg = None,
    token: TokenArg = None,
) -> None:
    """Install a marketplace model's verified stable version into a running CHAP deployment.

    Run from your CHAP Docker Compose deployment directory. The model template and its
    verified configurations are registered in CHAP from the marketplace entry, and the
    service is started. Custom chapkit images can be installed with --image IMAGE
    --accept-risk; they are registered from the running service. Docker Compose is required.
    """
    _deploy(
        model,
        compose_file,
        image,
        accept_risk,
        False,
        platform,
        updating=False,
        no_start=no_start,
        url=url,
        token=token,
    )


def update(
    model: ModelArg,
    *,
    compose_file: ComposeArg = (Path("compose.yml"),),
    image: ImageArg = None,
    accept_risk: RiskArg = False,
    platform: PlatformArg = None,
    no_start: NoStartArg = False,
    url: UrlArg = None,
    token: TokenArg = None,
) -> None:
    """Update an installed model to the marketplace's verified stable version.

    The new version is registered in CHAP as a new template version with its
    configurations; earlier versions and their backtests are untouched. Custom models
    require --accept-risk again. Use --image to select a new custom image, or omit it
    to pull the previously installed custom image reference.
    """
    _deploy(
        model, compose_file, image, accept_risk, False, platform, updating=True, no_start=no_start, url=url, token=token
    )


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
        _remove(model, compose_file, local=False, delete_data=delete_data, api=api)


def start(
    model: ModelArg,
    *,
    image: ImageArg = None,
    accept_risk: RiskArg = False,
    platform: PlatformArg = None,
) -> None:
    """Run a marketplace model as a standalone local service for chap eval. No CHAP server needed.

    Starts the model's verified stable version, or moves an already started model to
    it. The service gets a port on 127.0.0.1; the command prints the URL to pass to
    chap eval --model-name. Custom chapkit images need --image IMAGE --accept-risk.
    """
    _deploy(model, (), image, accept_risk, True, platform, updating=None)


def stop(model: ModelArg, *, delete_data: DeleteDataArg = False) -> None:
    """Stop and remove a local model service started with chap model start.

    The model's data volume is kept so a later start resumes from it; pass
    --delete-data to remove it permanently.
    """
    _remove(model, (), local=True, delete_data=delete_data, api=None)


def _remove(
    model: str, compose_file: tuple[Path, ...], *, local: bool, delete_data: bool, api: "ChapApi | None"
) -> None:
    import yaml

    from chap_core.log_config import initialize_logging

    initialize_logging()
    try:
        overlay, config, service_name = _prepare(model, compose_file, local)
        previous = config["services"].pop(service_name, None)
        if previous is None:
            raise ValueError(f"Model '{model}' is not installed.")
        if api is not None:
            _retire_template(api, previous.get("x-chap-template"))
        volume = f"{service_name}-data"
        config["volumes"].pop(volume, None)
        command = [*_compose_command(compose_file, local), "-f", str(overlay)]
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
    except (ValueError, OSError, yaml.YAMLError, subprocess.CalledProcessError) as error:
        logger.error("%s", error)
        raise SystemExit(1) from error


def _retire_template(api: "ChapApi", template_name: str | None) -> None:
    """Archive the live template the overlay recorded for the service, if CHAP still lists it."""
    if template_name is None:
        logger.info("The overlay records no model template for this service, so nothing is retired in CHAP.")
        return
    live = [template for template in api.model_templates() if template["name"] == template_name]
    if not live:
        logger.info("Model template %s is not listed by CHAP at %s, so nothing is retired.", template_name, api.url)
        return
    api.archive_model_template(live[0]["id"])
    logger.info("Retired model template %s and its configured models in CHAP at %s.", template_name, api.url)


def _prepare(model: str, compose_files: tuple[Path, ...], local: bool) -> tuple[Path, dict[str, Any], str]:
    """Validate the arguments and return the overlay path, its config and the service name."""
    import yaml

    if not re.fullmatch(r"[a-z0-9][a-z0-9_]*", model):
        raise ValueError("Model names must contain only lowercase letters, digits and underscores.")
    if not local and (not compose_files or any(not path.is_file() for path in compose_files)):
        raise ValueError("Run from a CHAP deployment directory or pass its base files with --compose-file.")
    if local:
        overlay = Path.home() / ".chap" / "compose.models.yml"
        overlay.parent.mkdir(parents=True, exist_ok=True)
    else:
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
    return overlay, config, f"marketplace-{model.replace('_', '-')}"


def _compose_command(compose_files: tuple[Path, ...], local: bool) -> list[str]:
    command = ["docker", "compose"]
    if local:
        command.extend(["--project-name", "chap-local-models"])
    else:
        for path in compose_files:
            command.extend(["-f", str(path.resolve())])
    return command


def _write_pending(config: dict[str, Any], overlay: Path) -> Path:
    """Write the new model Compose file beside the overlay it will replace."""
    import yaml

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


def _register_marketplace_model(api: "ChapApi", pin: "ModelPin") -> str:
    """Store the pin's template and verified configurations in CHAP. Returns the template name."""
    from chap_core.services.model_marketplace import configured_model_requests, model_template_request

    template = api.create_model_template(model_template_request(pin))
    logger.info(
        "Registered model template %s version %s (id %s) in CHAP at %s.",
        template["name"],
        template["version"],
        template["id"],
        api.url,
    )
    requests = configured_model_requests(pin, template["id"])
    for request in requests:
        api.create_configured_model(request)
    logger.info("Created %d configured models from the marketplace entry.", len(requests))
    return str(template["name"])


def _register_custom_model(api: "ChapApi", service_name: str) -> str:
    """Store the template a just started custom service describes. Returns the template name.

    The service's own id is only known once it has registered, so it is found by the
    hostname it registered from.
    """
    deadline = time.monotonic() + REGISTRATION_TIMEOUT
    while True:
        registered = [
            service for service in api.services() if service["url"].split("//", 1)[-1].split(":")[0] == service_name
        ]
        if registered:
            break
        if time.monotonic() > deadline:
            raise ValueError(
                f"Service {service_name} did not register with CHAP at {api.url} within "
                f"{REGISTRATION_TIMEOUT} seconds. Custom images must support chapkit self-registration."
            )
        time.sleep(2)
    template = api.create_model_template_from_service(registered[0]["id"])
    api.create_configured_model({"name": "default", "model_template_id": template["id"], "user_option_values": {}})
    logger.info(
        "Registered model template %s version %s (id %s) with a default configuration in CHAP at %s.",
        template["name"],
        template["version"],
        template["id"],
        api.url,
    )
    return str(template["name"])


def _deploy(
    model: str,
    compose_files: tuple[Path, ...],
    image: str | None,
    accept_risk: bool,
    local: bool,
    platform: str | None,
    updating: bool | None,
    no_start: bool = False,
    url: str | None = None,
    token: str | None = None,
) -> None:
    import httpx
    import yaml

    from chap_core.log_config import initialize_logging
    from chap_core.services.chap_api import ChapApi
    from chap_core.services.model_marketplace import DEFAULT_REGISTRY_URL, registry_url, resolve_model

    initialize_logging()
    api = None if local else ChapApi(url, token)
    try:
        overlay, config, service_name = _prepare(model, compose_files, local)
        services = config["services"]
        previous = services.get(service_name)
        if updating is None:
            updating = previous is not None
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
            if no_start:
                raise ValueError("A custom image is registered from the running service, so it cannot use --no-start.")
            image = image if image is not None else previous["image"]
            if not image or any(character.isspace() for character in image) or "$" in image:
                raise ValueError("Provide a valid custom container image reference.")
            version = "custom"
            pin = None
        else:
            pin = resolve_model(model, registry)
            image, version = pin.image, pin.version

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
            if local:
                del service["environment"]
                del service["depends_on"]
                service["ports"] = [{"target": 8000, "host_ip": "127.0.0.1"}]
        service.update({"image": image, "x-chap-custom": bool(custom), "x-chap-version": version})
        if not custom:
            service["x-chap-registry"] = registry
        if platform is not None:
            if not re.fullmatch(r"[a-z0-9][a-z0-9/._-]*", platform):
                raise ValueError("Provide a valid container platform, for example linux/amd64.")
            service["platform"] = platform
        services[service_name] = service

        # A marketplace model is registered before its container runs, so CHAP knows the
        # model even when the service is not started. Every call can be repeated.
        if api is not None and pin is not None:
            service["x-chap-template"] = _register_marketplace_model(api, pin)

        command = _compose_command(compose_files, local)
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
                    f" {_command_name(local, updating)} {model} --platform linux/amd64"
                )
            logger.error("Could not pull %s.%s", image, hint)
            raise SystemExit(1) from error
        if previous is None:
            # chapkit keeps its SQLite database under data/ in the image's working directory, which
            # the image owns for its service user; mounting elsewhere leaves a root-owned volume.
            workdir = (_inspect_image(image, "{{.Config.WorkingDir}}") or "/").rstrip("/")
            service["volumes"] = [f"{service_name}-data:{workdir}/data", {"type": "tmpfs", "target": "/tmp"}]
        # Publish the new pin only after Docker has started it successfully.
        pending = _write_pending(config, overlay)
        try:
            pending_command = [*command, "-f", str(pending)]
            up = ["up", "-d", "--no-deps", "--wait", "--wait-timeout", "120", service_name]
            if not no_start:
                try:
                    subprocess.run([*pending_command, *up], check=True)
                except (subprocess.CalledProcessError, KeyboardInterrupt):
                    if previous is not None:
                        logger.warning("Update failed; restoring the previous model service.")
                        if previous_image_id is not None:
                            subprocess.run(["docker", "tag", previous_image_id, previous["image"]], check=True)
                        subprocess.run([*command, "-f", str(overlay), *up], check=True)
                    else:
                        subprocess.run([*pending_command, "rm", "--stop", "--force", service_name], check=True)
                    raise
            if api is not None and custom:
                service["x-chap-template"] = _register_custom_model(api, service_name)
                pending.unlink()
                pending = _write_pending(config, overlay)
            pending.replace(overlay)
        finally:
            pending.unlink(missing_ok=True)
        logger.info(
            "%s %s (%s): %s%s",
            "Updated" if updating else "Installed",
            model,
            version,
            image,
            " (not started)" if no_start else "",
        )
        if local:
            result = subprocess.run(
                [*command, "-f", str(overlay), "port", service_name, "8000"],
                check=True,
                stdout=subprocess.PIPE,
                text=True,
            )
            logger.info(
                "Use with chap eval --model-name http://%s --run-config.is-chapkit-model", result.stdout.strip()
            )
        else:
            logger.info("Include -f %s in future Docker Compose commands for this deployment.", overlay)
    except (ValueError, OSError, httpx.HTTPError, yaml.YAMLError, subprocess.CalledProcessError) as error:
        logger.error("%s", error)
        raise SystemExit(1) from error
    finally:
        if api is not None:
            api.close()


def _command_name(local: bool, updating: bool) -> str:
    if local:
        return "chap model start"
    return f"chap-admin {'update' if updating else 'install'}"
