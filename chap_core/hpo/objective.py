import logging
from pathlib import Path

from chap_core.api_types import BackTestParams
from chap_core.assessment.evaluation import Evaluation
from chap_core.cli_endpoints.utils import export_metrics
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
    ):
        self.model_template = model_template
        self.backtest_params = backtest_params
        self.metric = metric
        self.historical_context_years = historical_context_years

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
        logger.debug(f"Including {self.historical_context_years} years of historical context for plotting")

        evaluation = Evaluation.create(
            configured_model=configured_model_db,
            estimator=estimator,
            dataset=dataset,
            backtest_params=self.backtest_params,
            backtest_name="hpo_evaluation",
            historical_context_years=self.historical_context_years,
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
