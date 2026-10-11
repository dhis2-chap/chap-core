"""
Launch an MLproject model as a chapkit service with ``chapkit mlproject run``.

chapkit itself runs on chap-core's interpreter. It runs the MLproject commands
as plain shell commands, so the model's environment is prepared here first and
then put in front of chapkit (``uv run``), or picked up by the commands
themselves (renv activates through the project's ``.Rprofile``).
"""

import importlib.metadata
import logging
import os
import shutil
import sys
from pathlib import Path

import yaml
from packaging.version import Version

from chap_core.exceptions import InvalidModelException
from chap_core.runners.command_line_runner import run_command

logger = logging.getLogger(__name__)

CHAPKIT_MLPROJECT_RUN = [sys.executable, "-m", "chapkit.cli.cli", "mlproject", "run", "."]
UNSUPPORTED_ENVS = ("docker_env", "conda_env", "python_env")
# First chapkit release whose `mlproject run` serves the full MLproject contract.
MIN_CHAPKIT_VERSION = Version("2.3.1")


def check_chapkit_version() -> None:
    """Fail early when the installed chapkit has no usable `mlproject run`."""
    try:
        installed = Version(importlib.metadata.version("chapkit"))
    except importlib.metadata.PackageNotFoundError:
        raise InvalidModelException(
            "chapkit is not installed in the environment running Chap; reinstall chap-core."
        ) from None
    if installed < MIN_CHAPKIT_VERSION:
        raise InvalidModelException(
            f"--run-config.as-chapkit needs chapkit >= {MIN_CHAPKIT_VERSION}, but {installed} is installed. "
            "Upgrade it, for example with `uv sync` in a chap-core checkout."
        )
    logger.info(f"Serving the model with chapkit {installed} (`chapkit mlproject run`)")


def _require_tool(tool: str, env_key: str) -> None:
    if shutil.which(tool) is None:
        raise InvalidModelException(
            f"The MLproject declares {env_key}, which needs `{tool}` on PATH, but it was not found. "
            "Install it, or use --run-config.ignore-environment to run the commands in the current environment."
        )


def chapkit_service_launch(mlproject_file: Path, ignore_env: bool = False) -> tuple[list[str], dict[str, str]]:
    """Prepare the model environment and return the command and environment that start the service.

    The returned command is run from the MLproject directory, with ``--port`` and
    ``--host`` appended by ``ChapkitServiceManager``.
    """
    working_dir = Path(mlproject_file).parent.resolve()
    with open(mlproject_file) as file:
        mlproject = yaml.safe_load(file) or {}

    env = dict(os.environ)
    # A local run must never register itself with an orchestrator.
    env.pop("SERVICEKIT_ORCHESTRATOR_URL", None)
    # Startup waits for uvicorn's INFO-level "Uvicorn running on" line, which a higher level would hide.
    if env.get("LOG_LEVEL", "INFO").upper() not in ("DEBUG", "INFO"):
        env["LOG_LEVEL"] = "INFO"

    if ignore_env:
        return list(CHAPKIT_MLPROJECT_RUN), env

    unsupported = [key for key in UNSUPPORTED_ENVS if mlproject.get(key) is not None]
    if unsupported:
        raise InvalidModelException(
            f"Running an MLproject with {unsupported[0]} as a chapkit service is not supported. "
            "Use --run-config.ignore-environment to run its commands in the current environment, "
            "or start a chapkit image yourself (for example "
            "docker run -p 8000:8000 -v $(pwd):/work ghcr.io/dhis2-chap/chapkit-r-inla-cli) "
            "and pass its URL as --model-name."
        )

    if mlproject.get("uv_env") is not None:
        _require_tool("uv", "uv_env")
        env["UV_PROJECT_ENVIRONMENT"] = str(working_dir / ".venv")
        logger.info(f"Syncing uv environment in {working_dir}")
        run_command("uv sync", working_dir, env=env)
        return ["uv", "run", "--no-sync", *CHAPKIT_MLPROJECT_RUN], env

    if mlproject.get("renv_env") is not None:
        _require_tool("Rscript", "renv_env")
        logger.info(f"Restoring renv environment in {working_dir}")
        run_command('Rscript -e "renv::restore(prompt = FALSE)"', working_dir, env=env)

    return list(CHAPKIT_MLPROJECT_RUN), env
