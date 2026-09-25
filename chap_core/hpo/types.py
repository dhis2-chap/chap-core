from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Literal

from .search_space import serialize_search_space

if TYPE_CHECKING:
    from pathlib import Path

    from chap_core.api_types import BacktestParams
    from chap_core.database.model_templates_and_config_tables import ModelConfiguration


HpoStopReason = Literal["max_trials", "search_exhausted"]


# Mainly for future adaptive parallel search to keep track of trial id/token
@dataclass(frozen=True, slots=True)
class SearchCandidate:
    params: dict[str, Any]
    token: int | None = None


@dataclass(frozen=True, slots=True)
class Trial:
    trial_nr: int  # reconstruct convergence curves with leaderboard
    params: dict[str, Any]
    score: float | None
    seconds: float
    failure: str | None


@dataclass(frozen=True)
class HyperparameterOptimization:
    """
    Runtime inputs and results of one HPO run.
    The search space and parameter dictionaries are independent copies.
    """

    # Inputs to HyperparameterOptimizer
    searcher: str
    model_template_name: str
    model_template_version: str
    backtest_params: BacktestParams
    metric: str
    search_space: dict[str, Any]
    max_trials: int | None
    seed: int | None

    # Results
    model_configuration: ModelConfiguration
    best_params: dict[str, Any]
    best_score: float
    leaderboard: list[Trial]

    # Execution metadata
    seconds: float
    stop_reason: HpoStopReason

    def to_flat(self) -> FlatHyperparameterOptimization:
        from chap_core.assessment.metrics import get_optimization_direction

        successful = sum(t.score is not None for t in self.leaderboard)
        return FlatHyperparameterOptimization(
            searcher=self.searcher,
            model_template_name=self.model_template_name,
            model_template_version=self.model_template_version,
            backtest_params=self.backtest_params.model_dump(),
            metric=self.metric,
            direction=get_optimization_direction(self.metric).value,
            search_space=serialize_search_space(self.search_space),
            max_trials=self.max_trials,
            seed=self.seed,
            model_configuration=self.model_configuration.model_dump(mode="json"),
            best_params=self.best_params,
            best_score=float(self.best_score),
            n_trials=len(self.leaderboard),
            n_successful_trials=successful,
            n_failed_trials=len(self.leaderboard) - successful,
            seconds=float(self.seconds),
            stop_reason=self.stop_reason,
        )

    def write_leaderboard(self, filepath: Path) -> None:
        param_names = sorted({param_name for trial in self.leaderboard for param_name in trial.params})
        fieldnames = [
            "trial_nr",
            *param_names,
            "score",
            "seconds",
            "failure",
        ]
        with filepath.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for trial in self.leaderboard:
                writer.writerow(
                    {
                        "trial_nr": trial.trial_nr,
                        **trial.params,
                        "score": trial.score,
                        "seconds": trial.seconds,
                        "failure": trial.failure,
                    }
                )

    def write_trials(self, filepath: Path) -> None:
        with filepath.open("w", encoding="utf-8") as f:
            for trial in sorted(self.leaderboard, key=lambda t: t.trial_nr):
                json.dump(asdict(trial), f)
                f.write("\n")


@dataclass(frozen=True, slots=True)
class FlatHyperparameterOptimization:
    """
    Serializable representation of an HPO run.
    """

    searcher: str
    model_template_name: str
    model_template_version: str
    backtest_params: dict[str, Any]
    metric: str
    direction: str
    search_space: dict[str, Any]
    max_trials: int | None
    seed: int | None
    model_configuration: dict[str, Any]
    best_params: dict[str, Any]
    best_score: float
    n_trials: int
    n_successful_trials: int
    n_failed_trials: int
    seconds: float
    stop_reason: HpoStopReason
