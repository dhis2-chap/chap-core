"""
Comparison operators for metrics.

A comparison operator turns a model's score and a reference model's score (for
example a baseline) into one number. All operators are oriented so that a
positive value means the model is better than the reference and zero means
they are equal.

Operators are registered by name with ``@comparison_op(name)``, and a metric
selects one through ``MetricSpec.comparison_op``.
"""

import math
from collections.abc import Callable
from typing import TYPE_CHECKING

from chap_core.assessment.metrics.base import OptimizationDirection, TargetBehavior

if TYPE_CHECKING:
    from chap_core.assessment.metrics.base import MetricSpec

ComparisonFn = Callable[[float, float, "MetricSpec"], float]

_comparison_registry: dict[str, ComparisonFn] = {}


def comparison_op(name: str):
    """Decorator to register a comparison operator under a name."""

    def decorator(fn: ComparisonFn) -> ComparisonFn:
        if name in _comparison_registry:
            raise ValueError(f"Comparison operator {name!r} is already registered")
        _comparison_registry[name] = fn
        return fn

    return decorator


def get_comparison_op(name: str) -> ComparisonFn:
    """Get a comparison operator by name."""
    if name not in _comparison_registry:
        raise ValueError(f"Unknown comparison operator {name!r}. Available: {sorted(_comparison_registry)}")
    return _comparison_registry[name]


def list_comparison_ops() -> list[str]:
    """List the names of all registered comparison operators."""
    return sorted(_comparison_registry)


def _require_direction(spec: "MetricSpec") -> OptimizationDirection:
    if spec.optimization_direction is None:
        raise ValueError(f"Metric {spec.metric_id!r} needs an optimization direction for this comparison")
    return spec.optimization_direction


@comparison_op("skill_ratio")
def skill_ratio(score: float, reference_score: float, spec: "MetricSpec") -> float:
    """Relative improvement over the reference: 0.2 means 20% better.

    For error scores (minimize) this is the skill score ``1 - score / reference``.
    Undefined (NaN) when the reference score is 0.
    """
    direction = _require_direction(spec)
    if reference_score == 0:
        return math.nan
    if direction == OptimizationDirection.MINIMIZE:
        return 1 - score / reference_score
    return score / reference_score - 1


@comparison_op("difference")
def difference(score: float, reference_score: float, spec: "MetricSpec") -> float:
    """Absolute improvement over the reference in the metric's own units."""
    direction = _require_direction(spec)
    if direction == OptimizationDirection.MINIMIZE:
        return reference_score - score
    return score - reference_score


def _distance_to_target(value: float, target: float, behavior: TargetBehavior) -> float:
    if behavior == TargetBehavior.AT_LEAST:
        return max(target - value, 0.0)
    return abs(value - target)


@comparison_op("target_distance")
def target_distance(score: float, reference_score: float, spec: "MetricSpec") -> float:
    """How much closer to the target the model is than the reference.

    With ``TargetBehavior.AT_LEAST``, scores at or above the target have distance 0.
    """
    if spec.target is None:
        raise ValueError(f"Metric {spec.metric_id!r} needs a target for this comparison")
    return _distance_to_target(reference_score, spec.target, spec.target_behavior) - _distance_to_target(
        score, spec.target, spec.target_behavior
    )
