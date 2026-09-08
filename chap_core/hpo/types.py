from dataclasses import dataclass
from typing import Any
from .base import write_yaml

from chap_core.database.base_tables import DBModel

DEFAULT_HPO_TRIALS = 3


# Mainly for future adaptive parallel searching to keep track of trial id/token
@dataclass(frozen=True, slots=True)
class SearchCandidate:
    params: dict[str, Any]
    token: int | None = None


class Trial(TypedDict): # LeaderboardEntry
    config: dict[str, Any]
    score: float


@dataclass(frozen=True)
class HpoRun(DBModel):
    # Persisting a reproducible run would additionally require serializable component settings and seeds.
    """
    Contains inputs and results from one hyperparameter optimization run.
    The objective and searcher are live references. The search space and parameter
    dictionaries are independent copies. 
    """

    # objective: Objective
    # searcher: Searcher
    # direction: OptimizationDirection
    # model_configuration: ModelConfiguration | None
    # best_params: dict[str, Any]
    # best_score: float
    # leaderboard: list[LeaderboardEntry]

    def write_best_config(self, output_yaml):
        if self._best_config is not None:
            write_yaml(output_yaml, self._best_config)