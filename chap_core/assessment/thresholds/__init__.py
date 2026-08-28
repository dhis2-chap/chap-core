"""Threshold (endemic channel) calculation plugin system.

Each strategy is a single class registered with the :func:`threshold` decorator,
mirroring the ``@backtest_plot`` registry. New strategies are added by writing a
class with a typed params model, importing its module from
:func:`_discover_strategies`, and adding the params model to
:data:`~chap_core.assessment.thresholds.params.ThresholdParams` — the endpoint
code never needs editing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from chap_core.assessment.thresholds.base import ThresholdStrategyBase

if TYPE_CHECKING:
    from pydantic import BaseModel

# Global registry for threshold strategies
_threshold_strategies_registry: dict[str, type[ThresholdStrategyBase]] = {}


def threshold(strategy_id: str, name: str, params_model: type[BaseModel], description: str = ""):
    """Decorator to register a threshold strategy class with its typed params model."""

    def decorator(cls: type[ThresholdStrategyBase]) -> type[ThresholdStrategyBase]:
        if not issubclass(cls, ThresholdStrategyBase):
            raise TypeError(f"{cls.__name__} must inherit from ThresholdStrategyBase")

        type_literal = params_model.model_fields["type"].default
        if type_literal != strategy_id:
            raise ValueError(
                f"{params_model.__name__}.type literal {type_literal!r} does not match "
                f"registered strategy id {strategy_id!r}"
            )

        cls.id = strategy_id
        cls.name = name
        cls.description = description
        cls.params_model = params_model

        _threshold_strategies_registry[strategy_id] = cls
        return cls

    return decorator


def get_threshold_strategies_registry() -> dict[str, type[ThresholdStrategyBase]]:
    return _threshold_strategies_registry.copy()


def get_threshold_strategy(strategy_id: str) -> type[ThresholdStrategyBase] | None:
    return _threshold_strategies_registry.get(strategy_id)


def list_threshold_strategies() -> list[dict]:
    return [
        {
            "id": cls.id,
            "name": cls.name,
            "description": cls.description,
        }
        for cls in _threshold_strategies_registry.values()
    ]


def _discover_strategies():
    from chap_core.assessment.thresholds import (
        percentile,
        seasonal,
    )


_discover_strategies()
