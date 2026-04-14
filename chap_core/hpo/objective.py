import logging
from pathlib import Path
from uuid import uuid4

from chap_core.api_types import BackTestParams
from chap_core.assessment.evaluation import Evaluation
from chap_core.cli_endpoints.utils import calculate_metrics
from chap_core.database.model_templates_and_config_tables import ModelConfiguration
from chap_core.models.model_template import ModelTemplate
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class Objective:
    def __init__(
        self,
        model_template: ModelTemplate,
        backtest_params: BackTestParams,
        metric: str = "rmse",
        historical_context_years: int = 6,
        save_file: bool = False,
    ):
        self.model_template = model_template
        self.backtest_params = backtest_params
        self.metric = metric
        self.historical_context_years = historical_context_years
        self.save_file = save_file

    def __call__(self, config, dataset: DataSet) -> float:
        """
        This method takes a concrete configuration produced by a Searcher,
        runs model evaluation, and returns a scalar score of the selected metric.
        dry_run and plot from eval_cmd are not currently included.
        """
        from chap_core.database.model_templates_and_config_tables import ConfiguredModelDB, ModelTemplateDB

        logger.info("Validating model configuration")
        model_configs = {"user_option_values": config}  # TODO: should prob be removed
        configuration = ModelConfiguration.model_validate(model_configs)

        model = self.model_template.get_model(configuration)  # type: ignore[arg-type]
        estimator = model()

        run_id = uuid4().hex[:8]  # short unique id like 'a1b2c3d4'

        model_template_db = ModelTemplateDB(
            id=self.model_template.model_template_config.name,
            name=self.model_template.model_template_config.name,
            version=self.model_template.model_template_config.version or "unknown",
        )

        configured_model_db = ConfiguredModelDB(
            id=f"cli_hpo_eval_{run_id}",
            model_template_id=model_template_db.id,
            model_template=model_template_db,
            configuration=configuration.model_dump() if configuration else {},
        )

        logger.info(
            f"Running backtest with {self.backtest_params.n_splits} splits, {self.backtest_params.n_periods} periods, stride {self.backtest_params.stride}"
        )
        logger.debug(f"Including {self.historical_context_years} years of historical context for plotting")

        try:
            evaluation = Evaluation.create(
                configured_model=configured_model_db,
                estimator=estimator,
                dataset=dataset,
                backtest_params=self.backtest_params,
                backtest_name=f"hpo_evaluation_{run_id}",
                historical_context_years=self.historical_context_years,
            )
        except Exception:
            logger.exception(f"Evaluation failed for configuration {config}")
            raise
        # Knut: except Exception as e
        # raise exception from e

        if self.save_file:
            eval_file = Path(f"./chap_core/hpo/hpo_eval_{run_id}.nc")

            logger.info(f"Exporting hpo evaluation to {eval_file}")
            evaluation.to_file(
                filepath=eval_file,
                model_name=f"hpo_config_{run_id}",
                model_configuration=configuration.model_dump() if configuration else {},
                model_version=self.model_template.model_template_config.version or "unknown",
            )
            logger.info(f"Evaluation complete. Results saved to {eval_file}")

        logger.info("Calculating metrics for objective evaluation")
        metrics_results = calculate_metrics(
            evaluation=evaluation,
            metric_ids=[self.metric],
        )
        logger.info(f"Metrics calculation complete. Results: {metrics_results}")

        from chap_core.assessment.metrics import available_metrics

        print("metrics list:", list(available_metrics.keys()))
        print("metrics results:", metrics_results)

        metric_value = metrics_results[self.metric]
        if metric_value is None:
            raise ValueError(f"Metric {self.metric} could not be calculated for this configuration.")
        return float(metric_value)
