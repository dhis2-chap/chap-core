"""Install and update chapkit model services in a CHAP Compose deployment."""

import logging
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Annotated

from cyclopts import Parameter

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
LocalArg = Annotated[
    bool, Parameter(help="Run a standalone local service for CLI evaluations (no CHAP server needed).")
]
PlatformArg = Annotated[str | None, Parameter(help="Container platform, e.g. linux/amd64 for R-INLA on Apple Silicon.")]


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
    from chap_core.services.model_marketplace import resolve_model

    initialize_logging()
    try:
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
        if not isinstance(config, dict) or not isinstance(config.get("services"), dict):
            raise ValueError(f"Invalid model Compose file: {overlay}")
        config.setdefault("volumes", {})
        service_name = f"marketplace-{model.replace('_', '-')}"
        services = config["services"]
        previous = services.get(service_name)
        if updating and previous is None:
            raise ValueError(f"Model '{model}' is not installed. Run 'chap install {model}' first.")
        if not updating and previous is not None:
            raise ValueError(f"Model '{model}' is already installed. Run 'chap update {model}' instead.")

        custom = image is not None or (previous is not None and previous.get("x-chap-custom", False))
        if custom:
            warning = (
                "Custom models are not reviewed by the CHAP marketplace. You accept responsibility "
                "for running their code, sharing data with them, and using their forecasts."
            )
            if not accept_risk:
                raise ValueError(f"{warning} Pass --accept-risk to continue.")
            logger.warning(warning)
            image = image if image is not None else previous["image"]
            if not image or any(character.isspace() for character in image) or "$" in image:
                raise ValueError("Provide a valid custom container image reference.")
            version = "custom"
        else:
            pin = resolve_model(model)
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
                "volumes": [f"{service_name}-data:/app/data", {"type": "tmpfs", "target": "/tmp"}],
                "depends_on": {"chap": {"condition": "service_healthy"}},
            }
            config["volumes"][f"{service_name}-data"] = {}
            if local:
                del service["environment"]
                del service["depends_on"]
                service["ports"] = [{"target": 8000, "host_ip": "127.0.0.1"}]
        service.update({"image": image, "x-chap-custom": bool(custom), "x-chap-version": version})
        if platform is not None:
            service["platform"] = platform
        services[service_name] = service

        command = ["docker", "compose"]
        if local:
            command.extend(["--project-name", "chap-local-models"])
        else:
            for path in compose_files:
                command.extend(["-f", str(path.resolve())])
        # Publish the new pin only after Docker has pulled and started it successfully.
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", dir=overlay.parent, delete=False) as temporary:
            pending = Path(temporary.name)
            yaml.safe_dump(config, temporary, sort_keys=False)
        try:
            pending_command = [*command, "-f", str(pending)]
            subprocess.run([*pending_command, "pull", service_name], check=True)
            up = ["up", "-d", "--no-deps", "--wait", "--wait-timeout", "120", service_name]
            try:
                subprocess.run([*pending_command, *up], check=True)
            except subprocess.CalledProcessError:
                if previous is not None:
                    logger.warning("Update failed; restoring the previous model service.")
                    subprocess.run([*command, "-f", str(overlay), *up], check=True)
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
                capture_output=True,
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
