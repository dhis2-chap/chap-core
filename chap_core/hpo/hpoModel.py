import logging
from typing import Literal, Optional, Any, Tuple

from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.database.model_templates_and_config_tables import ModelConfiguration
from chap_core.file_io.example_data_set import DataSetType

from .hpoModelInterface import HpoModelInterface
from .objective import Objective
from .searcher import Searcher
from .base import write_yaml

Direction = Literal["maximize", "minimize"]

Direction = Literal["maximize", "minimize"]


logger = logging.getLogger()
logger.setLevel(logging.INFO)


class HpoModel(HpoModelInterface):
    def __init__(
        self,
        searcher: Searcher,
        objective: Objective,
        direction: Direction = "minimize",
        # model_configuration_yaml: Optional[str] = None,
        model_configuration: Optional[dict[str, list]] = None,
    ):
        if direction not in ("maximize", "minimize"):
            raise ValueError("direction must be 'maximize' or 'minimize'")

        self._searcher = searcher
        self._objective = objective
        self._direction = direction
        # self._model_configuration_yaml = model_configuration_yaml #TODO: this should a parsed dict
        self.base_configs = model_configuration

    def train(
        self,
        dataset: Optional[DataSetType],
    ) -> Tuple[str, dict[str, Any]]:
        """
        Runs hyperparameter optimization over a discrete search space.
        Returns the optimized and trained (trained on the whole dataset argument) predictor.
        """
        # hpo_configs = self.base_configs["user_option_values"]
        # for key, vals in hpo_configs.items(): # wraps Float and Int in a list
        #     deduped = dedup(vals)
        #     if not deduped:
        #         raise ValueError(f"'user_option_values.{key}' has no values to try.")
        #     hpo_configs[key] = deduped
        # self.base_configs.pop("user_option_values")

        best_score = float("inf") if self._direction == "minimize" else float("-inf")
        best_params: dict[str, Any] = {}
        # best_config = None

        self._searcher.reset(self.base_configs)
        while True:
            params = self._searcher.ask()
            print(f"params from searcher: {params}")
            if params is None:
                break

            trial_number = None
            if params.get("_trial_id") is not None:  # for TPESearcher
                trial_number = params.pop("_trial_id")

            # config = self.base_configs.copy()
            # config["user_option_values"] = params

            # Maybe best to seperate hpo_config and other configs in two files ??
            score = self._objective(params, dataset)
            if trial_number is not None:
                params["_trial_id"] = trial_number
                self._searcher.tell(params, score)
                params.pop("_trial_id")
            else:
                self._searcher.tell(params, score)

            is_better = (score < best_score) if self._direction == "minimize" else (score > best_score)
            if is_better or best_params is None:
                best_score = score
                best_params = params
                # best_config = config # vs. model_config, safe_load vs. model_validate

            print(f"Tried {params} -> score={score}")

        best_config = {"user_option_values": best_params}
        self._best_config = best_config
        print(f"\nBest params: {best_params} | best score: {best_score}")

        template = self._objective.template
        # TODO: validate config without "user_option_values"
        if best_config is not None:
            logger.info(f"Validating best model configuration: {best_config}")
            config = ModelConfiguration.model_validate(best_config)
            logger.info(f"Validated best model configuration: {config}")
        estimator = template.get_model(best_config)
        self._predictor = estimator.train(dataset)
        return self._predictor

    def predict(self, historic_data: DataSet, future_data: DataSet) -> DataSet:
        self._predictor.predict(historic_data, future_data)

    @property
    def get_best_config(self):
        return self._best_config

    @property
    def write_best_config(self, output_yaml):
        if self._best_config is not None:
            write_yaml(output_yaml, self._best_config)
