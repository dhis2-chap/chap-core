import itertools
import math
import random
from typing import TYPE_CHECKING, Any

import optuna

from .base import Float, Int
from .types import SearchCandidate

if TYPE_CHECKING:
    from collections.abc import Iterator


class Searcher:
    """Abstract optimizer interface.

    Implementations should:
    - call `reset(space)` before use
    - repeatedly return configurations via `ask()` until None (no more work)
    - receive feedback via `tell(params, result)`
    """

    def reset(self, search_space: dict[str, Any], seed: int | None) -> None: ...
    def ask(self) -> SearchCandidate | None: ...
    def tell(self, candidate: SearchCandidate, result: float) -> None: ...


class GridSearcher(Searcher):
    def __init__(self) -> None:
        self._iterator: Iterator[tuple[Any, ...]] | None = None
        self.keys: list[str] = []

    def reset(self, search_space: dict[str, list], seed: int | None) -> None:
        del seed
        self.keys = list(search_space.keys())
        for value in search_space.values():
            if not isinstance(value, list):
                raise ValueError("GridSearcher only supports list-based search spaces")
        self._iterator = itertools.product(*search_space.values())

    def ask(self) -> SearchCandidate | None:
        if self._iterator is None:
            raise RuntimeError("GridSearch not initialized. Call reset.")
        try:
            values = next(self._iterator)
        except StopIteration:
            return None
        params = dict(zip(self.keys, values, strict=True))
        return SearchCandidate(params=params)

    def tell(self, candidate: SearchCandidate, result: float) -> None:
        # Grid search doesn't adapt, but we keep the hook for API symmetry.
        return


class RandomSearcher(Searcher):
    """
    Samples with replacement.
    """

    def reset(self, search_space: dict[str, Any], seed: int | None = None) -> None:
        self.search_space = _validate_search_space_extended(search_space)
        self.rng = random.Random(seed)
        self.keys = list(search_space.keys())

    def _sample_float(self, s: Float) -> float:
        if s.log:
            low_log, high_log = math.log(s.low), math.log(s.high)
            u = self.rng.uniform(low_log, high_log)
            return math.exp(u)

        if s.step is None:
            return self.rng.uniform(s.low, s.high)

        n_float = (s.high - s.low) / s.step
        n = math.floor(n_float + 1e-12)
        k = self.rng.randint(0, n)
        return s.low + k * s.step

    def _sample_int(self, s: Int) -> int:
        if s.log:
            low_log, high_log = math.log(s.low), math.log(s.high + 1)  # +1 allows high to be sampled bc floor
            u = self.rng.uniform(low_log, high_log)
            x = math.floor(math.exp(u))
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

    def ask(self) -> SearchCandidate:
        if self.rng is None:
            raise RuntimeError("RandomSearch not initialized. Call reset")
        params = {k: self._sample_one(self.search_space[k]) for k in self.keys}
        return SearchCandidate(params=params)

    def tell(self, candidate: SearchCandidate, result: float) -> None:
        # Random search doesn't adapt, but we keep the hook for API symmetry.
        return


class TPESearcher(Searcher):
    """
    Tree Parzen Estimator searcher using Optuna's ask/tell with native distributions.
    Supports:
    - list[...] -> CategoricalDistribution
    - Float(low, high, step=None|>0, log=bool) -> FloatDistribution
    - Int(low, high, step>1, log=bool) -> IntDistribution
    """

    def __init__(self, direction: str):
        if direction not in ("maximize", "minimize"):
            raise ValueError("Invalid optimization direction")
        self.direction = direction
        self._pending: dict[int, optuna.trial.Trial] = {}
        self._study: optuna.study.Study | None = None

    def reset(self, search_space: dict[str, Any], seed: int | None = None) -> None:
        search_space = _validate_search_space_extended(search_space)

        self._keys = list(search_space.keys())
        self._dists = {k: _to_optuna_distr(v) for k, v in search_space.items()}
        self._study = optuna.create_study(
            direction=self.direction,
            sampler=optuna.samplers.TPESampler(seed=seed),
        )
        self._pending.clear()

    def ask(self) -> SearchCandidate:
        if self._study is None:
            raise RuntimeError("TPESearcher not initialized. Call reset")

        trial = self._study.ask(fixed_distributions=self._dists)
        self._pending[trial.number] = trial

        return SearchCandidate(params=dict(trial.params), token=trial.number)

    def _pop_trial(self, candidate: SearchCandidate) -> optuna.trial.Trial:
        if candidate.token is None:
            raise ValueError("TPE candidate is missing its trial token")
        try:
            return self._pending.pop(candidate.token)
        except KeyError:
            raise KeyError(f"No pending TPE trial {candidate.token}") from None

    def tell(self, candidate: SearchCandidate, result: float) -> None:
        trial = self._pop_trial(candidate)

        assert self._study is not None
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
