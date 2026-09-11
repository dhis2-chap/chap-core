"""Outbreak model plugin system.

An outbreak model answers one question: standing at an origin period, how likely
is it that a location breaches its epidemic channel in a later target period?

Two kinds implement that. A *predictive* model reads forecast samples and takes
the fraction above the channel. A *persistence-of-anomaly* model reads no
forecast at all -- it compares the origin period's own observation against the
channel for that season, on the reasoning that a year running hot in June is
likely still running hot in November.

Both share the threshold machinery and differ only in where the signal comes
from, so they are directly comparable against the same target. Models register
with the :func:`outbreak_model` decorator, mirroring the ``@threshold`` and
``@metric`` registries.
"""

from __future__ import annotations

from chap_core.assessment.outbreak.base import OutbreakModelBase

# Global registry for outbreak models
_outbreak_models_registry: dict[str, type[OutbreakModelBase]] = {}


def outbreak_model(model_id: str, name: str, description: str = ""):
    """Decorator to register an outbreak model class."""

    def decorator(cls: type[OutbreakModelBase]) -> type[OutbreakModelBase]:
        if not issubclass(cls, OutbreakModelBase):
            raise TypeError(f"{cls.__name__} must inherit from OutbreakModelBase")

        cls.id = model_id
        cls.name = name
        cls.description = description

        _outbreak_models_registry[model_id] = cls
        return cls

    return decorator


def get_outbreak_models_registry() -> dict[str, type[OutbreakModelBase]]:
    return _outbreak_models_registry.copy()


def get_outbreak_model(model_id: str) -> type[OutbreakModelBase] | None:
    return _outbreak_models_registry.get(model_id)


def list_outbreak_models() -> list[dict]:
    return [
        {
            "id": cls.id,
            "name": cls.name,
            "description": cls.description,
        }
        for cls in _outbreak_models_registry.values()
    ]


def _discover_models():
    from chap_core.assessment.outbreak import (
        persistence,
        threshold_model,
    )


_discover_models()
