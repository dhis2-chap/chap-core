from typing import Any, Optional
import itertools
import random
import optuna
import math
from .base import Int, Float

_TRIAL_ID_KEY = "_trial_id"  # reserved key we inject into params


class Searcher:
    """Abstract optimizer interface.

    Implementations should:
    - call `reset(space)` before use
    - repeatedly return configurations via `ask()` until None (no more work)
    - receive feedback via `tell(params, result)`
    """

    def reset(self, space: Any) -> None: ...
    def ask(self) -> Optional[dict[str, Any]]: ...
    def tell(self, params: dict[str, Any], result: float) -> None: ...


class GridSearcher(Searcher):
    def __init__(self):
        self._iterator: dict[str, Any] = None

    def reset(self, search_space: dict[str, list]) -> None:
        self.keys = list(search_space.keys())
        self._iterator = itertools.product(*search_space.values())

    def ask(self) -> Optional[dict[str, Any]]:
        if self._iterator is None:
            raise RuntimeError("GridSearch not initialized. Call reset(params).")
        try:
            ne = next(self._iterator)
            print(f"SEARCHER params: {dict(zip(self.keys, ne))}")
            return dict(zip(self.keys, ne))
        except StopIteration:
            return None

    def tell(self, params: dict[str, Any], result: float) -> None:
        # Grid search doesn't adapt, but we keep the hook for API symmetry.
        return


class RandomSearcher(Searcher):
    """
    Samples with replacement max_trials number of configurations.
    """

    def __init__(self, max_trials: int):
        if not isinstance(max_trials, int) or max_trials <= 0:
            raise ValueError("max_trials must be a positive integer")
        self.max_trials = max_trials

    def reset(self, search_space: dict[str, Any], seed: Optional[int] = None) -> None:
        self.search_space = _validate_search_space_extended(search_space)
        print(f"randomSearcher search space in reset: {self.search_space}")
        self.rng = random.Random(seed)
        self.keys = list(search_space.keys())
        self.emitted = 0

    def _sample_float(self, s: Float) -> float:
        if s.log:
            low_log, high_log = math.log(s.low), math.log(s.high)
            u = self.rng.uniform(low_log, high_log)
            return math.exp(u)

        if s.step is None:
            return self.rng.uniform(s.low, s.high)

        n_float = (s.high - s.low) / s.step
        n = int(math.floor(n_float + 1e-12))
        k = self.rng.randint(0, n)
        return s.low + k * s.step

    def _sample_int(self, s: Int) -> int:
        if s.log:
            low_log, high_log = math.log(s.low), math.log(s.high + 1)  # +1 allows high to be sampled bc floor
            u = self.rng.uniform(low_log, high_log)
            x = int(math.floor(math.exp(u)))
            return max(s.low, min(x, s.high))  # floating-point edges issues

        if s.step == 1:
            return self.rng.randint(s.low, s.high)

        n = (s.high - s.low) // s.step
        k = self.rng.randint(0, n)
        return s.low + k * s.step

    def _sample_one(self, spec: Any) -> Any:
        if isinstance(spec, list):
            return self.rng.choice(spec)
        if isinstance(spec, Float):
            return self._sample_float(spec)
        if isinstance(spec, Int):
            return self._sample_int(spec)
        raise TypeError(f"Unsupported spec at runtime: {spec!r}")

    def ask(self) -> Optional[dict[str, Any]]:
        if self.rng is None:
            raise RuntimeError("RandomSearch not initialized. Call reset(search_space, seed)")
        if self.emitted >= self.max_trials:
            return None
        params = {k: self._sample_one(self.search_space[k]) for k in self.keys}
        # config = {k: self.rng.choice(self.search_space[k]) for k in self.keys}
        self.emitted += 1
        return params

    def tell(self, params: dict[str, Any], result: float) -> None:
        # Random search doesn't adapt, but we keep the hook for API symmetry.
        return


class TPESearcher(Searcher):
    """
    Tree Parzen Estimator.
    Parallel-safe TPE searcher using Optuna's ask/tell with native distributions.
    - ask() returns a params dict that includes a reserved '_trial_id'.
    - tell() extracts '_trial_id' from params to update the correct trial.
    Supports:
    - list[...] -> CategoricalDistribution
    - Float(low, high, step=None|>0, log=bool) -> FloatDistribution
    - Int(low, high, step>1, log=bool) -> IntDistribution
    """

    def __init__(self, direction: str = "minimize", max_trials: Optional[int] = None):
        if direction not in ("maximize", "minimize"):
            raise ValueError("direction must be 'maximize' or 'minimize'")
        self.direction = direction
        self.max_trials = max_trials
        self._pending: dict[int, optuna.trial.Trial] = {}
        self._study: Optional[optuna.study.Study] = None
        self._asked = 0

    def reset(self, search_space: dict[str, list], seed: Optional[int] = None) -> None:
        # validate_search_space(search_space)
        search_space = _validate_search_space_extended(search_space)

        self._keys = list(search_space.keys())
        self._dists = {
            k: _to_optuna_distr(v)
            for k, v in search_space.items()
            # k: optuna.distributions.CategoricalDistribution(tuple(search_space[k]))
            # for k in self._keys
        }
        self._study = optuna.create_study(
            direction=self.direction,
            sampler=optuna.samplers.TPESampler(seed=seed),
        )
        self._pending.clear()
        self._asked = 0

    def ask(self) -> Optional[dict[str, Any]]:
        if self._study is None:
            raise RuntimeError("TPESearcher not initialized. Call reset(search_space, seed)")

        if self.max_trials is not None and self._asked >= self.max_trials:
            return None

        trial = self._study.ask(self._dists)
        self._pending[trial.number] = trial
        self._asked += 1

        params = dict(trial.params)
        params[_TRIAL_ID_KEY] = trial.number
        return params

    def tell(self, params: dict[str, Any], result: float) -> None:
        if _TRIAL_ID_KEY not in params:
            raise ValueError(f"params must include '{_TRIAL_ID_KEY}' returned by ask()")

        trial_id = params[_TRIAL_ID_KEY]
        trial = self._pending.pop(trial_id, None)
        if trial is None:
            raise KeyError(f"No pending trial with id {trial_id}")

        self._study.tell(trial, result)


def validate_search_space(search_space: dict[str, list]):
    if not isinstance(search_space, dict) or not search_space:
        raise ValueError("search_space must be a non-empty dict[str, list]")
    for k, v in search_space.items():
        if not isinstance(v, list) or not v:
            raise ValueError(f"search_space['{k}'] must be a non-empty list; got {v!r}")


def _validate_search_space_extended(search_space: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(search_space, dict) or not search_space:
        raise ValueError("search_space must be a non-empty dict")

    normalized: dict[str, Any] = {}

    for k, spec in search_space.items():
        print(f"key, spec in validate_space: {k}, {spec}")
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
            else:
                if spec.step is not None:
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
                    raise ValueError(f"Int('{k}'): log=True requies low, high > 0")
            normalized[k] = Int(low=low, high=high, step=step, log=spec.log)
            continue

        raise ValueError(f"Unsupported spec for '{k}': expected list, Float, or Int; got {type(spec).__name__}")

    return normalized


def _to_optuna_distr(spec: Any):
    """
    Convert our spec to an Optuna Distribution.
    Supports: list (categorical), Float, Int.
    """
    if isinstance(spec, list):
        if not spec:
            raise ValueError("categorical list must be non-empty")
        return optuna.distributions.CategoricalDistribution(tuple(spec))

    if isinstance(spec, Float):
        return optuna.distributions.FloatDistribution(
            low=spec.low,
            high=spec.high,
            step=spec.step,
            log=spec.log,
        )

    if isinstance(spec, Int):
        return optuna.distributions.IntDistribution(
            low=spec.low,
            high=spec.high,
            step=spec.step,
            log=spec.log,
        )

    raise TypeError(f"Unsupported spec type: {type(spec).__name__}. Expected list, Float, or Int.")
