import logging
from typing import Literal

from chap_core.assessment.prediction_evaluator import evaluate_model
from chap_core.database.model_templates_and_config_tables import ModelConfiguration
from chap_core.exceptions import NoPredictionsError
from chap_core.file_io.example_data_set import DataSetType
from chap_core.models.model_template import ModelTemplate
from chap_core.assessment.evaluation import Evaluation
from chap_core.cli_endpoints.utils import export_metrics
from chap_core.api_types import BackTestParams
from pathlib import Path

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class Objective:
    def __init__(
        self,
        model_template: ModelTemplate,
        backtest_params: BackTestParams,
        metric: str = "MSE",
        # prediction_length: int = 3,  # 6,
        # n_splits: int = 4,
        ignore_environment: bool = False,
        debug: bool = False,
        log_file: str | None = None,
        run_directory_type: Literal["latest", "timestamp", "use_existing"] | None = "timestamp",
    ):
        self.model_template = model_template
        self.backtest_params = backtest_params
        self.metric = metric
        # self.prediction_length = prediction_length
        # self.n_splits = n_splits

    def __call__(self, config, dataset: DataSetType | None = None) -> float:
        """
        This method takes a concrete configuration produced by a Searcher,
        runs model evaluation, and returns a scalar score of the selected metric.
        """
        logger.info("Validating model configuration")
        model_configs = {"user_option_values": config}  # TODO: should prob be removed
        model_config = ModelConfiguration.model_validate(model_configs)
        logger.info("Validated model configuration")

        model = self.model_template.get_model(model_config)  # type: ignore[arg-type]
        model = model()
        try:
            # evaluate_model should handle CV/nested CV and return mean results
            # stratified fold/splits
            results = evaluate_model(
                model,
                dataset,  # type: ignore[arg-type]
                prediction_length=self.prediction_length,
                n_test_sets=self.n_splits,
            )
        except NoPredictionsError as e:
            logger.error(f"No predictions were made: {e}")
            return float("inf")
        return float(results[0][self.metric])
    
    def get_score(self, config, dataset: DataSetType | None = None) -> float:
        from chap_core.database.model_templates_and_config_tables import ConfiguredModelDB, ModelTemplateDB
        
        logger.info("Validating model configuration")
        model_configs = {"user_option_values": config}  # TODO: should prob be removed
        configuration = ModelConfiguration.model_validate(model_configs)

        model = self.model_template.get_model(configuration)  # type: ignore[arg-type]
        estimator = model()

        model_template_db = ModelTemplateDB(
            id=self.model_template.model_template_config.name,
            name=self.model_template.model_template_config.name,
            version=self.model_template.model_template_config.version or "unknown",
        )

        configured_model_db = ConfiguredModelDB(
            id="cli_hpo_eval",
            model_template_id=model_template_db.id,
            model_template=model_template_db,
            configuration=configuration.model_dump() if configuration else {},
        )

        logger.info(
            f"Running backtest with {self.backtest_params.n_splits} splits, {self.backtest_params.n_periods} periods, stride {self.backtest_params.stride}"
        )
        historical_context_years = 6
        evaluation = Evaluation.create(
            configured_model=configured_model_db,
            estimator=estimator,
            dataset=dataset,
            backtest_params=self.backtest_params,
            backtest_name="hpo_run",
            historical_context_years=historical_context_years,
        )

        output_file = Path("./chap_core/hpo/hpo_eval.nc")
        logger.info(f"Exporting evaluation to {output_file}")
        evaluation.to_file(
            filepath=output_file,
            model_name="hpo_config",
            model_configuration=configuration.model_dump() if configuration else {},
            model_version=self.model_template.model_template_config.version or "unknown",
        )
        logger.info(f"Evaluation complete. Results saved to {output_file}")

        metrics_results = export_metrics(
            input_files=[output_file],
            output_file=Path("./chap_core/hpo/metrics.csv"),
        )

        from chap_core.assessment.metrics import available_metrics
        print("metrics list:", list(available_metrics.keys()))
        return float(metrics_results[0][self.metric])