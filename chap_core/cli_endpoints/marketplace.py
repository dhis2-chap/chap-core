"""Install and update chapkit model services in a CHAP Compose deployment."""

import json
import logging
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Annotated, Any

from cyclopts import Parameter

logger = logging.getLogger(__name__)

DATA_MOUNT = "/app/data"
# chapkit images default DATABASE_URL to the relative path "data/chapkit.db", which resolves
# against each image's WORKDIR. Pin it to the mounted volume so a model whose WORKDIR is not
# /app does not try to create its database on the read-only root filesystem and fail to start.
DATABASE_URL = f"sqlite+aiosqlite:///{DATA_MOUNT}/chapkit.db"

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
LocalArg = Annotated[
    bool, Parameter(help="Run a standalone local service for CLI evaluations (no CHAP server needed).")
]
PlatformArg = Annotated[str | None, Parameter(help="Container platform, e.g. linux/amd64 for R-INLA on Apple Silicon.")]
DeleteDataArg = Annotated[bool, Parameter(help="Also delete the model's data volume. This cannot be undone.")]


def install(
    model: ModelArg,
    *,
    compose_file: ComposeArg = (Path("compose.yml"),),
    image: ImageArg = None,
    accept_risk: RiskArg = False,
    local: LocalArg = False,
    platform: PlatformArg = None,
) -> None:
    """Install a marketplace model's verified stable version into CHAP.

    Run from your CHAP Docker Compose deployment directory. Custom chapkit images
    can be installed with --image IMAGE --accept-risk. Docker Compose is required.
    """
    _deploy(model, compose_file, image, accept_risk, local, platform, updating=False)


def update(
    model: ModelArg,
    *,
    compose_file: ComposeArg = (Path("compose.yml"),),
    image: ImageArg = None,
    accept_risk: RiskArg = False,
    local: LocalArg = False,
    platform: PlatformArg = None,
) -> None:
    """Update an installed model to the marketplace's verified stable version.

    Custom models require --accept-risk again. Use --image to select a new custom
    image, or omit it to pull the previously installed custom image reference.
    """
    _deploy(model, compose_file, image, accept_risk, local, platform, updating=True)


def uninstall(
    model: ModelArg,
    *,
    compose_file: ComposeArg = (Path("compose.yml"),),
    local: LocalArg = False,
    delete_data: DeleteDataArg = False,
) -> None:
    """Remove an installed model service from CHAP.

    The model's data volume is kept so a later install resumes from it; pass
    --delete-data to remove it permanently. CHAP drops the service from its
    registry on its own once the container stops.
    """
    import yaml

    from chap_core.log_config import initialize_logging

    initialize_logging()
    try:
        overlay, config, service_name = _prepare(model, compose_file, local)
        if config["services"].pop(service_name, None) is None:
            raise ValueError(f"Model '{model}' is not installed.")
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


def _image_id(image: str) -> str | None:
    """Return the local image ID for a reference, so a rollback can restore a moved tag."""
    result = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image], stdout=subprocess.PIPE, text=True
    )
    return result.stdout.strip() if not result.returncode else None


def _restore(
    previous: dict[str, Any],
    previous_image_id: str | None,
    config: dict[str, Any],
    service_name: str,
    overlay: Path,
    command: list[str],
    up: list[str],
) -> None:
    """Start the previous service again, by image ID when the pull has moved its tag."""
    if previous_image_id is None:
        subprocess.run([*command, "-f", str(overlay), *up], check=True)
        return
    restored = {**config, "services": {**config["services"], service_name: {**previous, "image": previous_image_id}}}
    rollback = _write_pending(restored, overlay)
    try:
        subprocess.run([*command, "-f", str(rollback), *up], check=True)
    finally:
        rollback.unlink(missing_ok=True)


def _deploy(
    model: str,
    compose_files: tuple[Path, ...],
    image: str | None,
    accept_risk: bool,
    local: bool,
    platform: str | None,
    updating: bool,
) -> None:
    import httpx
    import yaml

    from chap_core.log_config import initialize_logging
    from chap_core.services.model_marketplace import DEFAULT_REGISTRY_URL, registry_url, resolve_model

    initialize_logging()
    try:
        overlay, config, service_name = _prepare(model, compose_files, local)
        services = config["services"]
        previous = services.get(service_name)
        if updating and previous is None:
            raise ValueError(f"Model '{model}' is not installed. Run 'chap install {model}' first.")
        if not updating and previous is not None:
            raise ValueError(f"Model '{model}' is already installed. Run 'chap update {model}' instead.")

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
                "volumes": [f"{service_name}-data:{DATA_MOUNT}", {"type": "tmpfs", "target": "/tmp"}],
                "depends_on": {"chap": {"condition": "service_healthy"}},
            }
            config["volumes"][f"{service_name}-data"] = {}
            if local:
                del service["environment"]
                del service["depends_on"]
                service["ports"] = [{"target": 8000, "host_ip": "127.0.0.1"}]
        # Applied on update too, so models installed before the pin existed receive it.
        environment = service.get("environment") or {}
        if isinstance(environment, list):
            if not any(str(entry).split("=")[0] == "DATABASE_URL" for entry in environment):
                environment = [*environment, f"DATABASE_URL={DATABASE_URL}"]
        else:
            environment = dict(environment)
            environment.setdefault("DATABASE_URL", DATABASE_URL)
        service["environment"] = environment
        service.update({"image": image, "x-chap-custom": bool(custom), "x-chap-version": version})
        if not custom:
            service["x-chap-registry"] = registry
        if platform is not None:
            if not re.fullmatch(r"[a-z0-9][a-z0-9/._-]*", platform):
                raise ValueError("Provide a valid container platform, for example linux/amd64.")
            service["platform"] = platform
        services[service_name] = service

        command = _compose_command(compose_files, local)
        previous_image_id = _image_id(previous["image"]) if previous is not None else None
        # Publish the new pin only after Docker has pulled and started it successfully.
        pending = _write_pending(config, overlay)
        try:
            pending_command = [*command, "-f", str(pending)]
            try:
                subprocess.run([*pending_command, "pull", service_name], check=True)
            except subprocess.CalledProcessError as error:
                # Docker has already printed why the pull failed, so add a hint instead of repeating it.
                hint = ""
                if platform is None and not service.get("platform"):
                    hint = (
                        " If this model publishes no image for your machine's architecture, retry with:"
                        f" chap {'update' if updating else 'install'} {model} --platform linux/amd64"
                    )
                logger.error("Could not pull %s.%s", image, hint)
                raise SystemExit(1) from error
            up = ["up", "-d", "--no-deps", "--wait", "--wait-timeout", "120", service_name]
            try:
                subprocess.run([*pending_command, *up], check=True)
            except (subprocess.CalledProcessError, KeyboardInterrupt):
                if previous is not None:
                    logger.warning("Update failed; restoring the previous model service.")
                    _restore(previous, previous_image_id, config, service_name, overlay, command, up)
                else:
                    subprocess.run([*pending_command, "rm", "--stop", "--force", service_name], check=True)
                raise
            pending.replace(overlay)
        finally:
            pending.unlink(missing_ok=True)
        logger.info("%s %s (%s): %s", "Updated" if updating else "Installed", model, version, image)
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


def register_commands(app):
    app.command()(install)
    app.command()(update)
    app.command()(uninstall)
