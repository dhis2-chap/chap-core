from typing import Literal, Optional

from chap_core.models.model_template import ModelTemplate
from chap_core.database.model_templates_and_config_tables import ModelConfiguration
from chap_core.assessment.prediction_evaluator import evaluate_model
from chap_core.exceptions import NoPredictionsError
from chap_core.file_io.example_data_set import DataSetType

import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class Objective:
    def __init__(
        self,
        model_template: ModelTemplate,
        metric: str = "MSE",
        prediction_length: int = 3,  # 6,
        n_splits: int = 4,
        ignore_environment: bool = False,
        debug: bool = False,
        log_file: Optional[str] = None,
        run_directory_type: Optional[Literal["latest", "timestamp", "use_existing"]] = "timestamp",
    ):
        self.model_template = model_template
        self.metric = metric
        self.prediction_length = prediction_length
        self.n_splits = n_splits

    def __call__(self, config, dataset: Optional[DataSetType] = None) -> float:
        """
        This method takes a concrete configuration produced by a Searcher,
        runs model evaluation, and returns a scalar score of the selected metric.
        """
        logger.info("Validating model configuration")
        model_configs = {"user_option_values": config}  # TODO: should prob be removed
        model_config = ModelConfiguration.model_validate(model_configs)
        logger.info("Validated model configuration")

        model = self.model_template.get_model(model_config)
        model = model()
        try:
            # evaluate_model should handle CV/nested CV and return mean results
            # stratified fold/splits
            results = evaluate_model(
                model,
                dataset,
                prediction_length=self.prediction_length,
                n_test_sets=self.n_splits,
            )
        except NoPredictionsError as e:
            logger.error(f"No predictions were made: {e}")
            return  # maybe return float("inf") here?
        return results[0][self.metric]
