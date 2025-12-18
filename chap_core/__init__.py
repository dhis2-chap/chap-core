"""Top-level package for chap-core."""

from pathlib import Path

from . import data, fetch
from .log_config import is_debug_mode
from .models.model_template_interface import ModelTemplateInterface

__author__ = """Chap Team"""
__email__ = "chap@dhis2.org"

# Read version from package metadata
try:
    from importlib.metadata import version as _get_version

    __version__ = _get_version("chap_core")
except Exception:
    __version__ = "unknown"

__minimum_modelling_app_version__ = "3.0.0"


def get_temp_dir() -> Path:
    """Get the temporary directory for build and test artifacts.

    Returns temporary directory path for storing build artifacts,
    test outputs, model files, and other temporary files.
    Creates the directory if it doesn't exist.

    Returns
    -------
    Path
        Path to the temporary directory (default: 'target/')
    """
    temp_dir = Path("target")
    temp_dir.mkdir(exist_ok=True)
    return temp_dir


__all__ = ["fetch", "data", "ModelTemplateInterface", "is_debug_mode", "get_temp_dir"]
