"""
Explainable AI (XAI) module for CHAP.

Provides model-agnostic explainability for probabilistic health forecasts.
"""

from .types import (
    FeatureAttribution,
    GlobalExplanation,
    LocalExplanation,
)

__all__ = [
    "FeatureAttribution",
    "GlobalExplanation",
    "LocalExplanation",
]
