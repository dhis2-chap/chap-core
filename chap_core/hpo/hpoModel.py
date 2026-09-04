import logging
from typing import Any, cast

from chap_core.database.model_templates_and_config_tables import ModelConfiguration
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

from .base import write_yaml
from .hpoModelInterface import HpoModelInterface
from .objective import Objective
from .searcher import Searcher

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class HpoModel(HpoModelInterface):
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
        self._configuration = configuration
        self._search_space = search_space
        self._max_trials = max_trials
        self._seed = seed
        self._best_config: dict[str, dict[str, Any]] | None = None
        self._leaderboard: list[dict[str, Any]] = []
        self._predictor: Any = None

    def train(self, dataset: DataSet) -> Any:  # type: ignore[override]
        """
        Calls get_leaderboard to find the optimal configuration.
        Then trains the tuned model on the whole input dataset (outer training set).
        """
        if self._best_config is None:
            logger.info("Running hyperparameter optimization to find the best configuration...")
            self._metadata = self.get_leaderboard(dataset)
        else:
            logger.info("Using previously found best configuration for retraining...")
        logger.info(f"Validating optimized model configuration: {self._best_config}")
        config = ModelConfiguration.model_validate(self._best_config)

        # updates the originial configuration for outer evaluation logging as long as user_option_values stays mutable
        # can also include additional_continuous_covariates if given in OG configuration for the optimized model below
        # if self._configuration is not None:
        #     self._configuration.user_option_values = dict(self._best_config["user_option_values"])
        # new object instead of overwrite
        # haven't decided on wether to merge

        template = self._objective.model_template
        estimator = template.get_model(config)  # type: ignore[arg-type]
        self._predictor = estimator.train(dataset)
        return self._predictor  # return self.....

    def predict(self, historic_data: DataSet[Any], future_data: DataSet[Any]) -> DataSet[Any]:
        assert self._predictor is not None, "Model not trained yet"
        return cast("DataSet[Any]", self._predictor.predict(historic_data, future_data))

    def get_leaderboard(self, dataset: DataSet) -> list[dict[str, Any]]:
        """
        Runs hyperparameter optimization over the search space.
        Returns a sorted list of configurations together with their score.
        """
        self._searcher.reset(self._search_space, self._seed)
        best_score: float | None = None
        best_params: dict[str, Any] | None = None
        trial_count = 0

        while True:
            if self._max_trials is not None and trial_count >= self._max_trials:
                break

            candidate = self._searcher.ask()
            if candidate is None:  # search exhausted
                break

            trial_count += 1
            score = self._objective(candidate.params, dataset)
            self._searcher.tell(candidate, score)

            self._leaderboard.append(
                {
                    "config": candidate.params,
                    "score": score,
                }
            )

            if self._is_better(score, best_score):
                best_score = score
                best_params = dict(candidate.params)

            logger.info(f"Validated {candidate.params} -> score={score}")

        if best_params is None or best_score is None:
            raise RuntimeError("HPO completed without any successful trials")
        self._best_config = {
            "user_option_values": best_params
        }  # this can be written to file and used as configuration for template
        logger.info(f"\nBest params: {best_params} | best score: {best_score}")
        self._leaderboard.sort(key=lambda conf: conf["score"], reverse=self._objective.direction.value == "maximize")
        assert best_params == self._leaderboard[0]["config"], "best params is not the first in leaderboard"
        return self._leaderboard

    def _is_better(self, score: float, incumbent: float | None) -> bool:
        if incumbent is None:
            return True
        if self._objective.direction.value == "minimize":
            return score < incumbent
        return score > incumbent

    @property
    def model_information(self):
        return self._objective.model_template.model_template_config

    @property
    def best_configuration(self):
        return self._best_config

    def write_best_config(self, output_yaml):
        if self._best_config is not None:
            write_yaml(output_yaml, self._best_config)
