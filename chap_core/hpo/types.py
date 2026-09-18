from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypedDict

from .base import Float, Int, write_yaml

if TYPE_CHECKING:
    from chap_core.database.model_templates_and_config_tables import ModelConfiguration
    from chap_core.hpo.objective import Objective
    from chap_core.hpo.searcher import Searcher

DEFAULT_HPO_TRIALS = (
    3  # 20, 50, 100 more reasonable, with 50 as the primary practical budget, 100 for convergence experiment.
)


# Mainly for future adaptive parallel searching to keep track of trial id/token
@dataclass(frozen=True, slots=True)
class SearchCandidate:
    params: dict[str, Any]
    token: int | None = None


class Trial(TypedDict):
    config: dict[str, Any]
    score: float


@dataclass(frozen=True)
class HyperparameterOptimization:  # Renamed to hpo instead of hpoRun since optimizer is only called once now
    """
    Runtime inputs and results of one HPO run.
    The objective and searcher are live references. The search space and parameter
    dictionaries are independent copies.
    """

    objective: Objective
    searcher: Searcher

    # Inputs required to reproduce the optimization
    search_space: dict[str, Any]
    max_trials: int | None
    seed: int | None

    # Results
    model_configuration: ModelConfiguration
    best_params: dict[str, Any]
    best_score: float
    leaderboard: list[Trial]

    def write_best_config(self, output_yaml):
        if self.model_configuration is not None:
            config = self.model_configuration.model_dump(mode="json")
            write_yaml(output_yaml, config)

    def to_flat(self) -> FlatHyperparameterOptimization:
        return FlatHyperparameterOptimization(
            searcher=type(self.searcher).__name__,
            model_template_name=self.objective.model_template.model_template_config.name,
            model_template_version=self.objective.model_template.model_template_config.version or "unknown",
            direction=self.objective.direction.value,
            metric=self.objective.metric,
            backtest_params=self.objective.backtest_params.model_dump(),
            search_space=serialize_search_space(self.search_space),
            max_trials=self.max_trials,
            seed=self.seed,
            model_configuration=self.model_configuration.model_dump(mode="json"),
            best_params=self.best_params,
            best_score=float(self.best_score),
            leaderboard=[
                {
                    "config": trial["config"],
                    "score": float(trial["score"]),
                }
                for trial in self.leaderboard
            ],
        )


@dataclass(frozen=True)
class FlatHyperparameterOptimization:  # or HpoRunMetadata
    """
    For persisting a reproducible run with serializable component settings and seeds.
    """

    searcher: str
    model_template_name: str
    model_template_version: str
    direction: str
    metric: str
    backtest_params: dict[str, Any]
    search_space: dict[str, Any]
    max_trials: int | None
    seed: int | None
    model_configuration: dict[str, Any]
    best_params: dict[str, Any]
    best_score: float
    leaderboard: list[dict[str, Any]]


# maybe put to .base
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
