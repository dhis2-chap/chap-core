"""Top-level package for chap-core."""

__author__ = """Chap Team"""
__email__ = "chap@dhis2.org"
__version__ = "1.1.0"
__minimum_modelling_app_version__ = "2.2.4"
from . import fetch
from . import data
from .models.model_template_interface import ModelTemplateInterface

__all__ = ["fetch", "data", "ModelTemplateInterface"]
