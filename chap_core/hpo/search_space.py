from dataclasses import dataclass
from typing import Any

"""
Constraints on non-categorical hyperparameter values:
    log=True
        floats: step=None and low/high > 0
        ints: step=1 and low/high > 0
    log=False (uniform)
        floats: step=None (continuous) or step > 0
        ints: step >= 1 and step-type=int
    Bounds are inclusive, when step is used, high is included only if it lies exactly on the grid.
"""


DEFAULT_HPO_TRIALS = (
    3  # 20, 50, 100 more reasonable, with 50 as the primary practical budget, 100 for convergence experiment.
)

# dataclass to pydantic later


@dataclass(frozen=True)
class Int:
    low: int
    high: int
    step: int = 1
    log: bool = False


@dataclass(frozen=True)
class Float:
    low: float
    high: float
    step: float | None = None
    log: bool = False


def search_space_from_config(config: dict) -> dict[str, Any]:
    space: dict[str, Any] = {}

    for name, spec in config.items():
        if not isinstance(spec, dict):
            raise ValueError(f"'{name}': each spec must be a mapping")

        # Categorical values
        if "values" in spec:
            values = spec["values"]
            if not isinstance(values, list) or not values:
                raise ValueError(f"'{name}': 'values' must be a non-empty list")
            space[name] = values
            continue

        if "low" not in spec or "high" not in spec:
            raise ValueError(f"'{name}': expected 'low' and 'high'")

        low, high = spec["low"], spec["high"]
        type_ = spec.get("type", None)  # default decided base on type of low, high
        log = bool(spec.get("log", False))
        step = spec.get("step", None)  # None for int is 1

        # Suggest int
        if (type_ or "").lower() == "int" or (isinstance(low, int) and isinstance(high, int) and type_ is None):
            if log and step not in (None, 1):
                raise ValueError(f"'{name}': log-int requires step==1 (or omit)")
            step_val = 1 if step is None else int(step)
            space[name] = Int(low=int(low), high=int(high), step=step_val, log=log)
            continue

        # Suggest float
        if (type_ or "").lower() == "float" or type_ is None:
            if log and step is not None:
                raise ValueError(f"'{name}': log-float requires step==None")
            float_step: float | None = None if step is None else float(step)
            space[name] = Float(low=float(low), high=float(high), step=float_step, log=log)
            continue

        raise ValueError(f"'{name}': unknown spec type '{type_}'")

    return space


def validate_search_space(search_space: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(search_space, dict) or not search_space:
        raise ValueError("search_space must be a non-empty dict")

    normalized: dict[str, Any] = {}

    for k, spec in search_space.items():
        # Categorical
        if isinstance(spec, list):
            if not spec:
                raise ValueError(f"list for '{k}' must be non-empty")
            normalized[k] = list(spec)
            continue

        # Suggest float
        if isinstance(spec, Float):
            low, high = float(spec.low), float(spec.high)
            if not (low < high):  # low != high
                raise ValueError(f"Float('{k}'): low < high required")
            if spec.log:
                if spec.step is not None:
                    raise ValueError(f"Float('{k}'): step must be None when log=True")
                if low <= 0 or high <= 0:
                    raise ValueError(f"Float('{k}'): log=True requires low, high > 0")
            elif spec.step is not None:
                if not (isinstance(spec.step, (int, float)) and spec.step > 0):
                    raise ValueError(f"Float('{k}'): step must be > 0")
            normalized[k] = Float(low=low, high=high, step=spec.step, log=spec.log)
            continue

        # Suggest int
        if isinstance(spec, Int):
            low, high, step = int(spec.low), int(spec.high), int(spec.step)
            if not (low <= high):
                raise ValueError(f"Int('{k}'): low <= high required")
            if step < 1:
                raise ValueError(f"Int('{k}'): step must be >= 1")
            if spec.log:
                if step != 1:
                    raise ValueError(f"Int('{k}'): step must be 1 when log=True")
                if low <= 0 or high <= 0:
                    raise ValueError(f"Int('{k}'): log=True requires low, high > 0")
            normalized[k] = Int(low=low, high=high, step=step, log=spec.log)
            continue

        raise ValueError(f"Unsupported spec for '{k}': expected list, Float, or Int; got {type(spec).__name__}")

    return normalized


def serialize_search_space(search_space: dict[str, Any]) -> dict[str, Any]:
    result = {}

    for name, value in search_space.items():
        if isinstance(value, Float):
            result[name] = {
                "type": "float",
                "low": value.low,
                "high": value.high,
                "step": value.step,
                "log": value.log,
            }
        elif isinstance(value, Int):
            result[name] = {
                "type": "int",
                "low": value.low,
                "high": value.high,
                "step": value.step,
                "log": value.log,
            }
        else:
            # categorical list
            result[name] = value

    return result
