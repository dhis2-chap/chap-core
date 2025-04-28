"""Top-level package for chap-core."""

__author__ = """Sandvelab"""
__email__ = "knutdrand@gmail.com"
__version__ = "1.0.7"

from . import fetch
from . import data
from .models.model_template_interface import ModelTemplateInterface

__all__ = ["fetch", "data", 'ModelTemplateInterface']
