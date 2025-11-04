# from dataclasses import dataclass, field
# from typing import Any, Dict, Optional, List, Protocol, runtime_checkable
# import time
import yaml
import json
from dataclasses import dataclass
from typing import Optional, Any


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
    step: Optional[float] = None
    log: bool = False


# maybe dataclass to pydantic later


def dedup(values):
    """Deduplicate while preserving order; works for scalars, lists, dicts, None."""
    seen = set()
    out = []
    for v in values if isinstance(values, list) else [values]:
        try:
            key = json.dumps(v, sort_keys=True, separators=(",", ":"), default=str)
        except Exception:
            key = repr(v)
        if key not in seen:
            seen.add(key)
            out.append(v)
    return out


def write_yaml(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def load_search_space_from_config(config: dict) -> dict[str, Any]:
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
            step_val = None if step is None else float(step)
            space[name] = Float(low=float(low), high=float(high), step=step_val, log=log)
            continue

        raise ValueError(f"'{name}': unknown spec type '{type_}'")

    print(f"space from load_yaml: {space}")
    return space


# @dataclass
# class ObjectiveResult:
#     """Standard result returned by an objective evaluation.

#     Attributes
#     ----------
#     score: float
#         The scalar to optimize. Higher is better if direction is "maximize".
#     metrics: Optional[Dict[str, float]]
#         Optional extra scalar metrics (loss, accuracy, etc.).
#     info: Optional[Dict[str, Any]]
#         Free-form info (e.g., training time). Not used by the optimizer.
#     """

#     score: float
#     metrics: Optional[Dict[str, float]] = None
#     info: Optional[Dict[str, Any]] = None


# @dataclass
# class Trial:
#     id: int
#     params: Dict[str, Any]
#     seed: int
#     started_at: float = field(default_factory=time.time)
#     ended_at: Optional[float] = None
#     result: Optional[ObjectiveResult] = None


# @dataclass
# class Study:
#     """Container for the full optimization run."""

#     trials: List[Trial]
#     direction: str # "maximize" or "minimize"
#     best_trial: Optional[Trial]
#     space: Any
#     optimizer: "Optimizer"

#     def best_score(self) -> Optional[float]:
#         return None if self.best_trial is None else self.best_trial.result.score # type: ignore[union-attr]
