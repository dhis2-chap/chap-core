"""Top-level package for chap-core."""

__author__ = """Chap Team"""
__email__ = "chap@dhis2.org"

# Read version from package metadata
try:
    from importlib.metadata import version as _get_version

    __version__ = _get_version("chap_core")
except Exception:
    __version__ = "unknown"

__minimum_modelling_app_version__ = "3.0.0"

from . import data, fetch
from .log_config import is_debug_mode
from .models.model_template_interface import ModelTemplateInterface

__all__ = ["fetch", "data", "ModelTemplateInterface", "is_debug_mode"]
