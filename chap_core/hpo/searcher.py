import itertools
import math
import random
from typing import TYPE_CHECKING, Any

import optuna

from .search_space import Float, Int, validate_search_space
from .types import SearchCandidate

if TYPE_CHECKING:
    from collections.abc import Iterator


class Searcher:
    """Searcher implementations must not mutate the supplied search_space.
    Otherwise defensive copies from caller needed.

    Implementations should:
    - call `reset(space)` before use
    - repeatedly return configurations via `ask()` until None (no more work)
    - receive feedback via `tell(params, result)`
    """

    def reset(self, search_space: dict[str, Any], seed: int | None) -> None: ...
    def ask(self) -> SearchCandidate | None: ...
    def tell(self, candidate: SearchCandidate, result: float | None) -> None: ...


class GridSearcher(Searcher):
    def __init__(self) -> None:
        self._iterator: Iterator[tuple[Any, ...]] | None = None
        self._keys: list[str] = []

    def reset(self, search_space: dict[str, Any], seed: int | None) -> None:
        del seed
        self._keys = list(search_space.keys())
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
        params = dict(zip(self._keys, values, strict=True))
        return SearchCandidate(params=params)

    def tell(self, candidate: SearchCandidate, result: float | None) -> None:
        # Grid search doesn't adapt, but we keep the hook for API symmetry.
        return


class RandomSearcher(Searcher):
    """Samples with replacement."""

    def __init__(self) -> None:
        self._search_space: dict[str, Any] = {}
        self._rng: random.Random | None = None
        self._keys: list[str] = []

    def reset(self, search_space: dict[str, Any], seed: int | None = None) -> None:
        self._search_space = validate_search_space(search_space)
        self._rng = random.Random(seed)
        self._keys = list(search_space.keys())

    def _sample_float(self, s: Float) -> float:
        assert self._rng is not None
        if s.log:
            low_log, high_log = math.log(s.low), math.log(s.high)
            u = self._rng.uniform(low_log, high_log)
            return math.exp(u)

        if s.step is None:
            return self._rng.uniform(s.low, s.high)

        n_float = (s.high - s.low) / s.step
        n = math.floor(n_float + 1e-12)
        k = self._rng.randint(0, n)
        return s.low + k * s.step

    def _sample_int(self, s: Int) -> int:
        assert self._rng is not None
        if s.log:
            low_log, high_log = math.log(s.low), math.log(s.high + 1)  # +1 allows high to be sampled bc floor
            u = self._rng.uniform(low_log, high_log)
            x = math.floor(math.exp(u))
            return max(s.low, min(x, s.high))  # floating-point edges issues

        if s.step == 1:
            return self._rng.randint(s.low, s.high)

        n = (s.high - s.low) // s.step
        k = self._rng.randint(0, n)
        return s.low + k * s.step

    def _sample_one(self, spec: Any) -> Any:
        assert self._rng is not None
        if isinstance(spec, list):
            return self._rng.choice(spec)
        if isinstance(spec, Float):
            return self._sample_float(spec)
        if isinstance(spec, Int):
            return self._sample_int(spec)
        raise TypeError(f"Unsupported spec at runtime: {spec!r}")

    def ask(self) -> SearchCandidate:
        assert self._rng is not None  # satisfy type checker
        # if self._rng is None:
        #     raise RuntimeError("RandomSearcher not initialized. Call reset")
        params = {k: self._sample_one(self._search_space[k]) for k in self._keys}
        return SearchCandidate(params=params)

    def tell(self, candidate: SearchCandidate, result: float | None) -> None:
        # Random search doesn't adapt to objective results.
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
        self._direction = direction
        self._pending: dict[int, optuna.trial.Trial] = {}
        self._study: optuna.study.Study | None = None

    def reset(self, search_space: dict[str, Any], seed: int | None = None) -> None:
        search_space = validate_search_space(search_space)

        self._keys = list(search_space.keys())
        self._dists = {k: _to_optuna_distr(v) for k, v in search_space.items()}
        self._study = optuna.create_study(
            direction=self._direction,
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

    def tell(self, candidate: SearchCandidate, result: float | None) -> None:
        trial = self._pop_trial(candidate)

        assert self._study is not None
        if result is None:
            self._study.tell(trial, state=optuna.trial.TrialState.FAIL)
        else:
            self._study.tell(trial, result)


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
