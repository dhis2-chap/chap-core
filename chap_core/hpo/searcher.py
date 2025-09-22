from typing import Any, Optional
import itertools
import random

import optuna

_TRIAL_ID_KEY = "_trial_id" # reserved key we inject into params


class Searcher():
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
    
    def reset(self, search_space: dict[str, list], seed: Optional[int] = None) -> None:
        if not isinstance(search_space, dict) or not search_space:
            raise ValueError("search_space must be a non-empty dict[str, list]")
        self.rng = random.Random(seed)
        self.search_space = search_space 
        self.keys = list(search_space.keys())
        self.emitted = 0

    def ask(self) -> Optional[dict[str, Any]]:
        if self.rng is None:
            raise RuntimeError("RandomSearch not initialized. Call reset(search_space, seed)")
        if self.emitted >= self.max_trials:
            return None 
        config = {k: self.rng.choice(self.search_space[k]) for k in self.keys}
        self.emitted += 1
        return config

    def tell(self, params: dict[str, Any], result: float) -> None:
        # Random search doesn't adapt, but we keep the hook for API symmetry.
        return 


# later for non-dicrete spaces too
class TPESearcher(Searcher):
    """
    Parallel-safe TPE searcher using Optuna's ask/tell interface.
    - ask() returns a params dict that includes a reserved '_trial_id'.
    - tell() extracts '_trial_id' from params to update the correct trial.
    """
    def __init__(self, direction: str = "minimize", max_trials: int = None):
        if direction not in ("maximize", "minimize"):
            raise ValueError("direction must be 'maximize' or 'minimize'")
        self.direction = direction 
        self.ax_trials = max_trials
        self._pending: dict[int, optuna.trial.Trial] = {}

    def reset(self, search_space: dict[str, list], seed: Optional[int] = None) -> None:
        validate_space(search_space)
        
        self._study = optuna.create_study(
            direction=self.direction,
            sampler=optuna.samplers.TPESampler(seed=seed),
        )

        self._keys = list(search_space.keys())
        self._dists = {
            k: optuna.distributions.CategoricalDistribution(tuple(search_space[k]))
            for k in self._keys
        }
        self._pending.clear()
        self._aksed = 0
    
    def ask(self) -> Optional[dict[str, Any]]:
        if self._study is None:
            raise RuntimeError("OptunaTPESearcher not initialized. Call reset(search_space, seed)")
        
        if self.max_trials is not None and self._aksed >= self.max_trials:
            return None 

        trial = self._study.ask(self._dists)
        self._pending[trial.number] = trial
        self._aksked += 1

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
        

def validate_space(search_space: dict[str, list]):
    if not isinstance(search_space, dict) or not search_space:
        raise ValueError("search_space must be a non-empty dict[str, list]")
    for k, v in search_space.items():
        if not isinstance(v, list) or not v:
            raise ValueError(f"search_space['{k}'] must be a non-empty list; got {v!r}")
        