import logging
from pathlib import Path

from chap_core.api_types import BacktestParams
from chap_core.assessment.evaluation import Evaluation
from chap_core.assessment.metrics import calculate_metrics
from chap_core.database.model_templates_and_config_tables import ModelConfiguration
from chap_core.models.model_template import ModelTemplate
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.util import generate_short_id

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class Objective:
    def __init__(
        self,
        model_template: ModelTemplate,
        backtest_params: BacktestParams,
        metric: str | None = None,
        historical_context_years: int = 6,
        eval_output_dir: Path | None = None,
    ):
        from chap_core.assessment.metrics import get_optimization_direction

        self.model_template = model_template
        self.backtest_params = backtest_params
        self.metric = metric or "rmse"
        self.direction = get_optimization_direction(self.metric)
        self.historical_context_years = historical_context_years
        self.eval_output_dir = eval_output_dir

    def __call__(self, params: dict, dataset: DataSet) -> float:
        """
        This method takes a concrete configuration produced by a Searcher,
        runs model validation, and returns a scalar score of the selected metric.
        """
        from chap_core.database.model_templates_and_config_tables import ConfiguredModelDB, ModelTemplateDB

        base_config = {"user_option_values": params}  # chap configuration file structure
        configuration = ModelConfiguration.model_validate(base_config)

        model = self.model_template.get_model(configuration)  # type: ignore[arg-type]
        estimator = model()

        run_id = generate_short_id()

        model_template_db = ModelTemplateDB(
            id=self.model_template.model_template_config.name,
            name=self.model_template.model_template_config.name,
            version=self.model_template.model_template_config.version or "unknown",
        )

        configured_model_db = ConfiguredModelDB(
            id=f"hpo_{run_id}",
            model_template_id=model_template_db.id,
            model_template=model_template_db,
            **configuration.model_dump() if configuration else {},
        )

        logger.info(
            f"Running validation backtest with {self.backtest_params.n_splits} splits, {self.backtest_params.n_periods} periods, stride {self.backtest_params.stride}"
        )
        logger.debug(f"Including {self.historical_context_years} years of historical context for plotting")

        try:
            evaluation = Evaluation.create(
                configured_model=configured_model_db,
                estimator=estimator,
                dataset=dataset,
                backtest_params=self.backtest_params,
                backtest_name=f"hpo_validation_{run_id}",
                historical_context_years=self.historical_context_years,
            )
        except Exception:
            logger.exception(f"Validation failed for configuration {base_config}")
            raise

        if self.eval_output_dir is not None:
            self.eval_output_dir.mkdir(parents=True, exist_ok=True)
            eval_file = self.eval_output_dir / f"hpo_validation_{run_id}.nc"

            logger.info(f"Exporting hpo validation to {eval_file}")
            evaluation.to_file(
                filepath=eval_file,
                model_name=f"hpo_config_{run_id}",
                model_configuration=configuration.model_dump() if configuration else {},
                model_version=self.model_template.model_template_config.version or "unknown",
            )
            logger.info(f"Validation complete. Results saved to {eval_file}")

        logger.info("Calculating metrics for objective validation")
        metrics = calculate_metrics(
            evaluation=evaluation,
            metric_ids=[self.metric],
        )
        logger.info(f"Metrics calculation complete: {metrics}")

        score = metrics[self.metric]
        if score is None:
            raise ValueError(f"Metric {self.metric} could not be calculated for this configuration.")
        return float(score)
