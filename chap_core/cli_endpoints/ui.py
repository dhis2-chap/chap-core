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
    "Install them with one of:\n"
    "  uv sync --extra ui\n"
    "  uv tool install 'chap_core[ui]'"
)


def ui_cmd(
    port: Annotated[int, Parameter(help="Port the UI listens on.")] = 8501,
    workdir: Annotated[Path, Parameter(help="Directory where uploads and evaluation runs are stored.")] = Path.home()
    / ".chap"
    / "ui",
    open_browser: Annotated[bool, Parameter(help="Open the UI in a browser on start.")] = True,
):
    """Start a local web UI for running and comparing model evaluations.

    The UI runs the same `chap eval` command under the hood and shows the
    equivalent CLI command for every run.
    """
    if importlib.util.find_spec("streamlit") is None:
        print(INSTALL_HINT)
        sys.exit(1)

    workdir.mkdir(parents=True, exist_ok=True)
    ui_dir = Path(__file__).parent.parent / "ui"
    app_path = ui_dir / "app.py"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
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
    env = {**os.environ, "CHAP_UI_WORKDIR": str(workdir.resolve())}
    sys.exit(subprocess.call(cmd, env=env))


def register_commands(app):
    """Register the ui command with the CLI app."""
    app.command(name="ui")(ui_cmd)
