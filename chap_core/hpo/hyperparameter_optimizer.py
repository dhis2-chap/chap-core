import logging
from copy import deepcopy
from typing import Any

from chap_core.database.model_templates_and_config_tables import ModelConfiguration
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

from .meta_learner import MetaLearner
from .objective import Objective
from .searcher import RandomSearcher, Searcher, TPESearcher
from .types import HyperparameterOptimization, Trial

logger = logging.getLogger()
logger.setLevel(logging.INFO)


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
        configuration: ModelConfiguration | None,
        search_space: dict[str, Any],
        max_trials: int | None,
        seed: int | None,
    ):
        self._objective = objective
        self._searcher = searcher
        self._base_config = deepcopy(configuration)
        self._search_space = search_space
        if max_trials is None and isinstance(searcher, (RandomSearcher, TPESearcher)):
            raise ValueError(
                f"max_trials must be specified for non-exhaustive searchers such as {type(searcher).__name__}"
            )
        self._max_trials = max_trials
        self._seed = seed

    def meta_learn(self, dataset: DataSet) -> HyperparameterOptimization:
        # model_configuration = deepcopy(self._base_config) # check for necessity of deepcopies, also this is not used is it necassary later only use self._base_config
        leaderboard: list[Trial] = []
        self._searcher.reset(deepcopy(self._search_space), self._seed)
        trial_count = 0

        while True:
            if self._max_trials is not None and trial_count >= self._max_trials:
                break

            candidate = self._searcher.ask()
            if candidate is None:  # search exhausted
                break
            params = deepcopy(candidate.params)

            trial_count += 1
            score = self._objective(deepcopy(params), dataset)
            self._searcher.tell(candidate, score)

            leaderboard.append(
                {
                    "config": params,
                    "score": score,
                }
            )
            logger.info(f"Tried {params} -> score={score}")

        if not leaderboard:
            raise ValueError("Hyperparameter optimization completed without any successful trials")

        leaderboard.sort(key=lambda conf: conf["score"], reverse=self._objective.direction.value == "maximize")
        best_candidate = leaderboard[0]
        logger.info("Best params: %s | best score: %s", best_candidate["config"], best_candidate["score"])
        best_model_config = {"user_option_values": best_candidate["config"]}
        # does not overwrite outer ModelTemplateDB.configuration since its dumped before hand
        # this includes additional_continuous_covariates if given in input configuration.yaml for the optimized model below
        if self._base_config is not None:
            self._base_config.user_option_values = best_model_config[
                "user_option_values"
            ]  # base_config has been deepcopied, does not overwrite
            logger.warning(
                "The original configuration has been updated with the best hyperparameter values found during optimization. "
                "The original additional_continuous_covariates will be preserved if they were present in the original configuration."
            )
        # configuration.user_option_values = {
        #     **(configuration.user_option_values or {}),
        #     **deepcopy(best_candidate["config"]),
        # }
        configuration = ModelConfiguration.model_validate(self._base_config or best_model_config)

        # template = self._objective.model_template
        # estimator = template.get_model(self._configuration if self._configuration is not None else config)  # type: ignore[arg-type]
        return HyperparameterOptimization(
            objective=self._objective,
            searcher=self._searcher,
            search_space=deepcopy(self._search_space),
            max_trials=self._max_trials,
            seed=self._seed,
            model_configuration=configuration,  # check needed
            best_params=deepcopy(best_candidate["config"]),
            best_score=best_candidate["score"],
            leaderboard=leaderboard,
        )

    @property
    def model_information(self):
        return self._objective.model_template.model_template_config
