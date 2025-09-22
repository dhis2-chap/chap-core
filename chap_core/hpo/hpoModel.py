<<<<<<< HEAD
from idlelib.debugobj import ObjectTreeItem
=======
>>>>>>> bb9fca5 (Composition, hpoModel, objective, searcher)
from typing import Literal, Optional, Any, Tuple, Callable
import yaml

from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.database.model_templates_and_config_tables import ModelConfiguration
from chap_core.file_io.example_data_set import DataSetType

from .hpoModelInterface import HpoModelInterface
from .searcher import Searcher
<<<<<<< HEAD
from .base import dedup, write_yaml
Direction = Literal["maximize", "minimize"]
=======
from .base import dedup, write_yaml 

Direction = Literal["maximize", "minimize"]

>>>>>>> bb9fca5 (Composition, hpoModel, objective, searcher)
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class HpoModel(HpoModelInterface):
    def __init__(
            self, 
            searcher: Searcher, 
<<<<<<< HEAD
            objective: 'Objective',
=======
            objective: callable, 
>>>>>>> bb9fca5 (Composition, hpoModel, objective, searcher)
            direction: Direction = "minimize", 
            model_configuration_yaml: Optional[str] = None,
    ):
        if direction not in ("maximize", "minimize"):
            raise ValueError("direction must be 'maximize' or 'minimize'")
<<<<<<< HEAD

        self._searcher = searcher
        self._objective = objective
        self._direction = direction
        self._model_configuration_yaml = model_configuration_yaml #TODO: this should a parsed dict
=======
        self._searcher = searcher
        self._objective = objective
        self._direction = direction
        self._model_configuration_yaml = model_configuration_yaml
>>>>>>> bb9fca5 (Composition, hpoModel, objective, searcher)
    
    def train(self, dataset: Optional[DataSetType],) -> Tuple[str, dict[str, Any]]:
        """
        Runs hyperparameter optimization over a discrete search space.
        Returns the optimized and trained (trained on the whole dataset argument) predictor.
        """
        if self._model_configuration_yaml is not None:
            logger.info(f"Loading model configuration from yaml file {self._model_configuration_yaml}")
            with open(self._model_configuration_yaml, "r", encoding="utf-8") as f:
                base_configs = yaml.safe_load(f) or {} # check if this returns a dict
            logger.info(f"Loaded model base configurations from yaml file: {base_configs}")

        if "user_option_values" not in base_configs or not isinstance(base_configs["user_option_values"], dict):
            raise ValueError("Expected top-level key 'user_option_values' mapping to a dict of lists.")

        hpo_configs = base_configs["user_option_values"]
        for key, vals in hpo_configs.items():
            deduped = dedup(vals)
            if not deduped:
                raise ValueError(f"'user_option_values.{key}' has no values to try.")
            hpo_configs[key] = deduped

        base_configs.pop("user_option_values")

        best_score = float("inf") if self._direction=="minimize" else float("-inf")
        best_params: dict[str, Any] = {}
        best_config = None

        self._searcher.reset(hpo_configs)
        while True:
            params = self._searcher.ask()
            if params is None:
                break 
<<<<<<< HEAD


            config = base_configs.copy()
            config["user_option_values"] = params

            # Maybe best to seperate hpo_config and other configs in two files ??
=======
            config = base_configs.copy()
            config["user_option_values"] = params

>>>>>>> bb9fca5 (Composition, hpoModel, objective, searcher)
            score = self._objective(config, dataset)
            self._searcher.tell(params, score)

            is_better = (score < best_score) if self._direction == "minimize" else (score > best_score)
            if is_better or best_params is None:
                best_score = score
                best_params = params
                best_config = config # vs. model_config, safe_load vs. model_validate

            print(f"Tried {params} -> score={score}")

        self._best_config = best_config
        print(f"\nBest params: {best_params} | best score: {best_score}")
        
        template = self._objective.template
        if best_config is not None:
            logger.info(f"Validating best model configuration: {best_config}")
            config = ModelConfiguration.model_validate(best_config)  
            logger.info(f"Validated best model configuration: {config}")
        estimator = template.get_model(config)
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