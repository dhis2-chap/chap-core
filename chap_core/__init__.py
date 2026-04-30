"""Top-level package for chap-core."""

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

# gluonts.json emits a UserWarning at import when neither orjson nor ujson
# is installed; chap_core does not exercise gluonts JSON I/O paths, so the
# warning is benign noise. Filter must be installed before any gluonts
# import happens transitively (CLIM-649).
warnings.filterwarnings(
    "ignore",
    message=r"Using `json`-module for json-handling",
    category=UserWarning,
    module=r"gluonts\.json",
)

if TYPE_CHECKING:
    # Re-imported under TYPE_CHECKING so static type checkers / IDEs still
    # see these public names. At runtime they are loaded lazily via
    # __getattr__ below to keep `import chap_core` cheap (CLIM-631).
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


def get_temp_dir() -> Path:
    """Get the temporary directory for build and test artifacts.

    Returns temporary directory path for storing build artifacts,
    test outputs, model files, and other temporary files.
    Creates the directory if it doesn't exist.

    Returns
    -------
    Path
        Path to the temporary directory (default: '/tmp/chap/temp/')
    """
    temp_dir = Path("/tmp/chap/temp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


# Lazy attribute loading (PEP 562). Keeps `import chap_core` cheap; the
# heavy submodules (data, fetch, model_template_interface) are loaded
# only on first access.
_LAZY_SUBMODULES = ("data", "fetch")
_LAZY_FROM = {
    "is_debug_mode": "chap_core.log_config",
    "ModelTemplateInterface": "chap_core.models.model_template_interface",
}


def __getattr__(name: str):
    import importlib

    if name in _LAZY_SUBMODULES:
        value = importlib.import_module(f"chap_core.{name}")
    elif name in _LAZY_FROM:
        value = getattr(importlib.import_module(_LAZY_FROM[name]), name)
    else:
        raise AttributeError(f"module 'chap_core' has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY_SUBMODULES, *_LAZY_FROM})


__all__ = ["ModelTemplateInterface", "data", "fetch", "get_temp_dir", "is_debug_mode"]
