"""Outbreak metric plugin system.

Outbreak metrics judge forecasts scored against the levels of an alert policy, see
:mod:`chap_core.assessment.outbreak_metrics.scoring`. They come in two kinds, each
with its own registry:

- **binary** metrics judge one level, as a function of its
  :class:`BinaryConfusion`. Register with :func:`binary_outbreak_metric`.
- **categorical** metrics judge the policy as a whole, as a function of the
  ``K x K`` confusion matrix of observed (rows) against predicted (columns)
  categories, where category ``0`` is no level and ``i + 1`` is level ``i``.
  Register with :func:`categorical_outbreak_metric`.

A metric returns ``None`` where it is undefined, e.g. sensitivity without outbreaks.
New metrics are added by writing a decorated function in :mod:`.binary` or
:mod:`.categorical`; the endpoint code never needs editing.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class BinaryConfusion(BaseModel):
    """Confusion counts of one alert level over a set of cells."""

    model_config = ConfigDict(frozen=True)

    tp: int = Field(ge=0, description="Outbreaks that were alerted.")
    fp: int = Field(ge=0, description="Alerts without an outbreak.")
    fn: int = Field(ge=0, description="Outbreaks that were missed.")
    tn: int = Field(ge=0, description="Quiet cells left quiet.")


BinaryMetricFn = Callable[[BinaryConfusion], float | None]
CategoricalMetricFn = Callable[[np.ndarray], float | None]


class OutbreakMetricKind(StrEnum):
    BINARY = "binary"
    CATEGORICAL = "categorical"


@dataclass(frozen=True)
class OutbreakMetricSpec[F]:
    id: str
    name: str
    description: str
    kind: OutbreakMetricKind
    compute: F


_binary_registry: dict[str, OutbreakMetricSpec[BinaryMetricFn]] = {}
_categorical_registry: dict[str, OutbreakMetricSpec[CategoricalMetricFn]] = {}


def binary_outbreak_metric(metric_id: str, name: str, description: str = ""):
    """Register a per-level metric computed from a :class:`BinaryConfusion`."""

    def decorator(fn: BinaryMetricFn) -> BinaryMetricFn:
        _binary_registry[metric_id] = OutbreakMetricSpec(metric_id, name, description, OutbreakMetricKind.BINARY, fn)
        return fn

    return decorator


def categorical_outbreak_metric(metric_id: str, name: str, description: str = ""):
    """Register a whole-policy metric computed from the observed x predicted confusion matrix."""

    def decorator(fn: CategoricalMetricFn) -> CategoricalMetricFn:
        _categorical_registry[metric_id] = OutbreakMetricSpec(
            metric_id, name, description, OutbreakMetricKind.CATEGORICAL, fn
        )
        return fn

    return decorator


def get_binary_outbreak_metrics() -> dict[str, OutbreakMetricSpec[BinaryMetricFn]]:
    return _binary_registry.copy()


def get_categorical_outbreak_metrics() -> dict[str, OutbreakMetricSpec[CategoricalMetricFn]]:
    return _categorical_registry.copy()


def _discover_metrics():
    from chap_core.assessment.outbreak_metrics import (
        binary,
        categorical,
    )


_discover_metrics()
