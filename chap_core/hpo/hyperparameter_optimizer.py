import logging
import math
from copy import deepcopy
from time import perf_counter
from typing import Any

from chap_core.database.model_templates_and_config_tables import ModelConfiguration
from chap_core.models.model_template import ModelTemplate
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

from .meta_learner import MetaLearner
from .objective import Objective
from .searcher import RandomSearcher, Searcher, TPESearcher
from .types import HpoStopReason, HyperparameterOptimization, Trial

logger = logging.getLogger(__name__)


class HyperparameterOptimizer(MetaLearner):
    """
    A HyperparameterOptimizer is a specififc implementation of a MetaLearner.
    It similar to configured models also contains a model template which can be
    accessed through its objective.
    """

    def __init__(
        self,
        *,
        objective: Objective,
        searcher: Searcher,
        model_configuration: ModelConfiguration | None,
        search_space: dict[str, Any],
        max_trials: int | None,
        seed: int | None,
    ):
        self._objective = objective
        self._searcher = searcher
        self._base_config = (
            model_configuration.model_copy(deep=True)  # ownership boundary
            if model_configuration is not None
            else None
        )
        self._search_space = deepcopy(search_space)  # ownership boundary
        if max_trials is None and isinstance(searcher, (RandomSearcher, TPESearcher)):
            raise ValueError(
                f"max_trials must be specified for non-exhaustive searchers such as {type(searcher).__name__}"
            )
        self._max_trials = max_trials
        self._seed = seed

    def meta_learn(self, dataset: DataSet) -> HyperparameterOptimization:
        hpo_start = perf_counter()  # benchmark performance counter
        leaderboard: list[Trial] = []
        self._searcher.reset(self._search_space, self._seed)
        trial_count = 0

        while True:
            if self._max_trials is not None and trial_count >= self._max_trials:
                stop_reason: HpoStopReason = "max_trials"
                break

            candidate = self._searcher.ask()
            if candidate is None:
                stop_reason = "search_exhausted"
                break
            params = dict(candidate.params)

            # each objective evaluation should include additional_continuous_covariates if inputed
            objective_config = self._make_configuration(params)

            trial_nr = trial_count
            trial_count += 1

            trial_start = perf_counter()
            score: float | None = None
            failure: str | None = None
            try:  # does trial failure first get caught here, does everyone earlier only raise it
                score = float(
                    self._objective(objective_config, dataset)
                )  # is constant floating needed, missing two places if needed
                if not math.isfinite(score):
                    raise ValueError(f"Objective returned non-finite score: {score}")
            except Exception as exc:
                score = None
                failure = f"{type(exc).__name__}: {exc}"
                self._searcher.tell(candidate, None)
                logger.exception(
                    "HPO trial %d failed for objective configuration %s",
                    trial_nr,
                    objective_config.model_dump(),
                )
            else:
                self._searcher.tell(candidate, score)

            trial_seconds = perf_counter() - trial_start

            leaderboard.append(
                Trial(
                    trial_nr=trial_nr,
                    params=params,
                    score=score,
                    seconds=trial_seconds,
                    failure=failure,
                )
            )
            if failure is None:
                logger.info(
                    "Trial %d: %s -> score=%s (%.3fs)",
                    trial_nr,
                    params,
                    score,
                    trial_seconds,
                )

        # loop end
        successful_trials = [trial for trial in leaderboard if trial.score is not None]

        if not successful_trials:
            raise ValueError("Hyperparameter optimization completed without any successful trials")

        successful_trials.sort(
            key=lambda trial: trial.score,  # type: ignore[arg-type, return-value]
            reverse=self._objective.direction.value == "maximize",
        )
        failed_trials = [trial for trial in leaderboard if trial.score is None]
        # successful trials ranked by score, failed trials afterwards.
        leaderboard = successful_trials + failed_trials

        best_candidate = leaderboard[0]
        best_score = best_candidate.score
        assert best_score is not None
        logger.info("Best params: %s | best score: %s", best_candidate.params, best_score)
        best_params = dict(best_candidate.params)
        # includes additional_continuous_covariates if given in input configuration.yaml
        best_configuration = self._make_configuration(best_params)
        if self._base_config is not None:
            logger.warning(
                "The best hyperparameter values found during optimization has been merged with the original model configuration. "
                "The original user_option_values will be preserved if they were not present in the search space. "
                "The original additional_continuous_covariates will be preserved if they were present in the original model configuration."
            )

        return HyperparameterOptimization(
            searcher=type(self._searcher).__name__,
            model_template_name=self._objective.model_template.model_template_config.name,
            model_template_version=self._objective.model_template.model_template_config.version or "unknown",
            backtest_params=self._objective.backtest_params,
            metric=self._objective.metric,
            search_space=deepcopy(self._search_space),
            max_trials=self._max_trials,
            seed=self._seed,
            model_configuration=best_configuration,  # check needed
            best_params=best_params,
            best_score=best_score,
            leaderboard=leaderboard,
            seconds=perf_counter() - hpo_start,
            stop_reason=stop_reason,
        )

    def _make_configuration(self, params: dict[str, Any]) -> ModelConfiguration:
        if self._base_config is None:
            return ModelConfiguration(
                user_option_values=dict(params),
            )

        config = deepcopy(self._base_config)
        # this replaces all existing user options in input yaml,
        # config.user_option_values = dict(params)
        # if preserve hyperparameters that were not part of search space use this
        config.user_option_values = {
            **(config.user_option_values or {}),
            **params,
        }
        return config

    @property
    def model_template(self) -> ModelTemplate:
        return self._objective.model_template

    @property
    def model_information(self):
        return self._objective.model_template.model_template_config
