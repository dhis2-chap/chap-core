"""Command that starts the local Streamlit frontend for the CHAP CLI."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import Annotated

from cyclopts import Parameter

INSTALL_HINT = (
    "The chap UI needs the optional 'ui' dependencies (streamlit).\n"
    "Run it with them, or install them, with one of:\n"
    "  uvx --from 'chap-core[ui]' chap ui\n"
    "  uv tool install 'chap-core[ui]'\n"
    "  uv sync --extra ui            (in a chap-core checkout)"
)

# The UI manages its runs with POSIX process groups and file locks. chap is used through WSL on Windows.
WINDOWS_HINT = "chap ui runs on macOS and Linux. On Windows, run it inside WSL, as the rest of chap."


def ui_cmd(
    port: Annotated[int, Parameter(help="Port the UI listens on.")] = 8501,
    host: Annotated[
        str,
        Parameter(
            help="Address the UI listens on. The default only accepts connections from this machine; "
            "0.0.0.0 lets anyone who can reach it run models, as the UI has no login."
        ),
    ] = "127.0.0.1",
    runs_dir: Annotated[
        Path,
        Parameter(
            help="Folder for runs, uploads, saved configurations and the models' working folders. "
            "The same folder chap eval uses (CHAP_RUNS_DIR)."
        ),
    ] = Path(os.environ.get("CHAP_RUNS_DIR", "runs")),
    uploads_dir: Annotated[
        Path | None, Parameter(help="Folder for files added through the browser. Default: uploads/ in the runs folder.")
    ] = None,
    models: Annotated[
        Path | None,
        Parameter(
            help="YAML list of your models, for example one shared by a team. Default: models.yaml in the runs folder."
        ),
    ] = None,
    chaps_project: Annotated[
        Path | None,
        Parameter(
            help="A chaps deployment whose models the UI should use and start. "
            "Found automatically when chap ui is started inside one."
        ),
    ] = None,
    registry_url: Annotated[
        str | None,
        Parameter(
            help="Model marketplace to list and start models from, for a fork or a mirror: its base URL "
            "or its registry.yaml. Passed on to chaps too."
        ),
    ] = None,
    open_browser: Annotated[bool, Parameter(help="Open the UI in a browser on start.")] = True,
):
    """Start a local web UI for running and comparing model evaluations.

    The UI runs the same chap commands under the hood and shows the equivalent
    CLI command for every run. Nothing is written until you run something,
    upload a file or save a configuration.
    """
    if sys.platform == "win32":
        print(WINDOWS_HINT)
        sys.exit(1)
    if importlib.util.find_spec("streamlit") is None:
        print(INSTALL_HINT)
        sys.exit(1)

    ui_dir = Path(__file__).parent.parent / "ui"
    app_path = ui_dir / "app.py"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.address",
        host,
        "--server.port",
        str(port),
        "--server.headless",
        str(not open_browser).lower(),
        "--browser.gatherUsageStats",
        "false",
        "--theme.base",
        str(ui_dir / "theme.toml"),
        "--client.toolbarMode",
        "minimal",
    ]
    settings = {
        "CHAP_RUNS_DIR": runs_dir,
        "CHAP_UPLOADS_DIR": uploads_dir,
        "CHAP_MODELS_FILE": models,
        "CHAPS_PROJECT_DIR": chaps_project,
    }
    env = {**os.environ, **{name: str(Path(value).resolve()) for name, value in settings.items() if value}}
    if registry_url:
        # chap-core's marketplace client takes the registry's base URL; accept the index file too.
        env["CHAP_MARKETPLACE_URL"] = registry_url.rstrip("/").removesuffix("/registry.yaml")
    sys.exit(subprocess.call(cmd, env=env))


def register_commands(app):
    """Register the ui command with the CLI app."""
    app.command(name="ui")(ui_cmd)
