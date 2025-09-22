# from dataclasses import dataclass, field
# from typing import Any, Dict, Optional, List, Protocol, runtime_checkable
# import time
import yaml
import json


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