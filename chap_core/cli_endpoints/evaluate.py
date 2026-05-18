"""Evaluation commands for CHAP CLI."""

from __future__ import annotations

import logging
from pathlib import Path  # noqa: TC003 — used at runtime via cyclopts get_type_hints()
from typing import TYPE_CHECKING, Annotated

from cyclopts import Parameter

from chap_core.api_types import BacktestParams, EstimatorMode, EstimatorOptions, RunConfig
from chap_core.cli_endpoints._args import (  # noqa: TC001 — used at runtime via cyclopts get_type_hints()
    BacktestParamsArg,
    DatasetCsvArg,
    DataSourceMappingArg,
    ModelConfigYamlArg,
    ModelNameArg,
    RunConfigArg,
)
from chap_core.cli_endpoints._common import (
    discover_geojson,
    get_estimator,
    get_hpo_estimator,
    load_dataset_from_csv,
    resolve_csv_path,
    warn_unused_covariates,
)

if TYPE_CHECKING:
    from chap_core.external.ExtendedPredictor import ExtendedPredictor
    from chap_core.hpo.hpoModel import HpoModel
    from chap_core.models.external_model import ExternalModel

logger = logging.getLogger(__name__)


def eval_cmd(
    model_name: ModelNameArg,
    dataset_csv: DatasetCsvArg,
    output_file: Annotated[
        Path,
        Parameter(help="Path for output NetCDF file containing evaluation results (.nc extension)"),
    ],
    backtest_params: BacktestParamsArg = BacktestParams(n_periods=3, n_splits=7, stride=1),
    run_config: RunConfigArg = RunConfig(),
    model_configuration_yaml: ModelConfigYamlArg = None,
    historical_context_years: Annotated[
        int,
        Parameter(help="Years of historical data to include for plotting context."),
    ] = 6,
    data_source_mapping: DataSourceMappingArg = None,
    dry_run: Annotated[bool, Parameter(help="Write inputs and print commands without executing.")] = False,
    plot: Annotated[bool, Parameter(help="Generate an HTML evaluation plot alongside the NetCDF.")] = False,
    estimator_options: Annotated[
        EstimatorOptions | None,
        Parameter(help="Estimator behavior (normal | hpo | ensemble), optional metric."),
    ] = None,
):
    """Evaluate a model using backtesting and export results to NetCDF format.

    Thin wrapper around :func:`_run_eval` that optionally records the run to
    MLflow when ``run_config.track`` is True. See :func:`_run_eval` for the
    detailed backtest workflow.
    """
    from chap_core.assessment.eval_tracking import load_model_configuration, tracked_eval_run

    model_configuration = load_model_configuration(model_configuration_yaml)

    with tracked_eval_run(
        track=run_config.track,
        model_name=model_name,
        dataset_csv=str(dataset_csv),
        backtest_params=backtest_params,
        historical_context_years=historical_context_years,
        model_configuration=model_configuration,
    ) as tracker:
        _run_eval(
            model_name=model_name,
            dataset_csv=dataset_csv,
            output_file=output_file,
            backtest_params=backtest_params,
            run_config=run_config,
            model_configuration_yaml=model_configuration_yaml,
            historical_context_years=historical_context_years,
            data_source_mapping=data_source_mapping,
            dry_run=dry_run,
            plot=plot,
            estimator_options=estimator_options,
        )
        tracker.log_outputs_from_files(
            output_file,
            plot_path=output_file.with_suffix(".html") if plot else None,
        )


def _run_eval(
    model_name: ModelNameArg,
    dataset_csv: DatasetCsvArg,
    output_file: Annotated[
        Path,
        Parameter(help="Path for output NetCDF file containing evaluation results (.nc extension)"),
    ],
    backtest_params: BacktestParamsArg = BacktestParams(n_periods=3, n_splits=7, stride=1),
    run_config: RunConfigArg = RunConfig(),
    model_configuration_yaml: ModelConfigYamlArg = None,
    historical_context_years: Annotated[
        int,
        Parameter(
            help="Years of historical data to include for plotting context. "
            "Calculated as periods based on dataset frequency (e.g., 6 years = 312 weeks or 72 months)"
        ),
    ] = 6,
    data_source_mapping: DataSourceMappingArg = None,
    dry_run: Annotated[
        bool,
        Parameter(
            help="Write data files and print commands without executing the model. "
            "Useful for debugging model inputs and verifying command formatting."
        ),
    ] = False,
    plot: Annotated[
        bool,
        Parameter(help="Generate an evaluation plot (HTML) alongside the NetCDF output"),
    ] = False,
    estimator_options: Annotated[
        EstimatorOptions | None,
        Parameter(
            help="Estimator behavior. Leave mode unset for a normal evaluation run. "
            "Use --estimator-options.mode=hpo for hyperparameter optimization. "
            "Use --estimator-options.mode=ensemble for ensemble learning. "
            "Optionally --estimator-options.metric=<metric> for hpo and ensemble."
        ),
    ] = None,
):
    """Evaluate a model using backtesting and export results to NetCDF format.

    Runs a rolling-origin backtest evaluation on a disease prediction model and saves
    results in NetCDF format for analysis with scientific tools. GeoJSON polygon files
    are auto-discovered from files with the same name as the CSV but with .geojson extension.

    The evaluation splits historical data into multiple train/test sets, trains the model
    on each training set, and generates probabilistic forecasts that are compared against
    actual observations. Results include predictions, observations, and computed metrics.

    HPO can be activated through estimator_options.mode, which will run a hyperparameter
    optimization over the search space defined in the model template or in the provided
    model_configuration_yaml file. The best configuration is selected based on the specified
    estimator_options.metric.

    Examples:
        # Evaluate a GitHub-hosted model
        chap eval --model-name https://github.com/dhis2-chap/minimalist_example_uv \\
            --dataset-csv ./data/vietnam.csv --output-file ./results/eval.nc

        # Evaluate a chapkit model (REST API)
        chap eval --model-name http://localhost:8000 --run-config.is-chapkit-model \\
            --dataset-csv ./data/vietnam.csv --output-file ./results/eval.nc

        # Use column name mapping when CSV columns don't match model expectations
        chap eval --model-name ./my_model --dataset-csv ./data.csv \\
            --output-file ./eval.nc --data-source-mapping ./column_mapping.json

        # Evaluate with hyperparameter optimization
        chap eval --model-name https://github.com/dhis2-chap/minimalist_example \\
            --dataset-csv ./example_data/vietnam_monthly.csv --output-file ./chap_core/hpo/eval.nc \\
            --model-configuration-yaml ./chap_core/hpo/config3.yaml --estimator-options.mode hpo \\
            --estimator_options.metric sensitivity
    """
    from chap_core.assessment.evaluation import Evaluation
    from chap_core.database.model_templates_and_config_tables import ConfiguredModelDB, ModelTemplateDB
    from chap_core.external.ExtendedPredictor import ExtendedPredictor
    from chap_core.hpo.searcher import RandomSearcher
    from chap_core.log_config import initialize_logging
    from chap_core.models.model_template import ModelTemplate
    from chap_core.models.utils import CHAP_RUNS_DIR

    # The same can be done for backtest_params and run_config,
    # or have them depend on cyclopts
    if estimator_options is None:
        estimator_options = EstimatorOptions()

    logger.info(f"Evaluating model {model_name} with xarray/NetCDF output")

    initialize_logging(run_config.debug, run_config.log_file)

    column_mapping = None
    if data_source_mapping is not None:
        import json

        logger.info(f"Loading column mapping from {data_source_mapping}")
        with open(data_source_mapping) as f:
            column_mapping = json.load(f)

    csv_path, url_geojson_path = resolve_csv_path(dataset_csv)
    geojson_path = url_geojson_path or discover_geojson(csv_path)
    dataset = load_dataset_from_csv(csv_path, geojson_path, column_mapping)

    if dry_run and estimator_options.mode != EstimatorMode.NORMAL:
        logger.warning(
            "Dry run does not support estimator_options.mode=%s; forcing mode='normal'.", estimator_options.mode.value
        )
        estimator_options = EstimatorOptions(mode=EstimatorMode.NORMAL, metric=estimator_options.metric)

    logger.info(f"Loading model template from {model_name}")
    template = ModelTemplate.from_directory_or_github_url(
        model_name,
        base_working_dir=CHAP_RUNS_DIR,
        ignore_env=run_config.ignore_environment,
        run_dir_type=run_config.run_directory_type,
        is_chapkit_model=run_config.is_chapkit_model,
        dry_run=dry_run,
    )

    configuration = None
    with template:
        estimator: ExternalModel | HpoModel | ExtendedPredictor
        if estimator_options.mode == EstimatorMode.NORMAL:
            estimator, configuration = get_estimator(template, model_configuration_yaml)
            warn_unused_covariates(dataset, template.model_template_config, configuration)
        elif estimator_options.mode == EstimatorMode.HPO:
            assert estimator_options.metric is not None
            estimator = get_hpo_estimator(
                template=template,
                model_configuration_yaml=model_configuration_yaml,
                backtest_params=backtest_params,
                metric=estimator_options.metric,
                searcher=RandomSearcher(2),
            )
        elif estimator_options.mode == EstimatorMode.ENSEMBLE:
            raise NotImplementedError("Ensemble mode is not yet implemented")

        model_info = estimator.model_information
        if model_info.min_prediction_length is None and model_info.max_prediction_length is None:
            logger.warning("Model has not specified minimum and maximum predicted length")
        if (
            model_info.min_prediction_length is not None
            and model_info.min_prediction_length > backtest_params.n_periods
        ):
            raise ValueError(
                f"The desired prediction length of {backtest_params.n_periods} is less than the model's minimum prediction length of {model_info.min_prediction_length}"
            )
        if (
            model_info.max_prediction_length is not None
            and model_info.max_prediction_length < backtest_params.n_periods
        ):
            logger.warning(
                f"Wrapping model to extend prediction length from {model_info.max_prediction_length} to {backtest_params.n_periods}. This is done iteratively, and may worsen model performance"
            )
            estimator = ExtendedPredictor(estimator, backtest_params.n_periods)

        model_template_db = ModelTemplateDB(
            id=template.model_template_config.name,
            name=template.model_template_config.name,
            version=template.model_template_config.version or "unknown",
        )

        configured_model_db = ConfiguredModelDB(
            id="cli_eval",
            model_template_id=model_template_db.id,
            model_template=model_template_db,
            configuration=configuration.model_dump() if configuration else {},
        )

        logger.info(
            f"Running backtest with {backtest_params.n_splits} splits, {backtest_params.n_periods} periods, stride {backtest_params.stride}"
        )
        logger.debug(f"Including {historical_context_years} years of historical context for plotting")

        if dry_run:
            from chap_core.assessment.prediction_evaluator import backtest

            for _ in backtest(
                estimator, dataset, backtest_params.n_periods, backtest_params.n_splits, backtest_params.stride
            ):
                pass
            return

        evaluation = Evaluation.create(
            configured_model=configured_model_db,
            estimator=estimator,
            dataset=dataset,
            backtest_params=backtest_params,
            backtest_name=f"{model_name}_evaluation",
            historical_context_years=historical_context_years,
        )

        logger.info(f"Exporting evaluation to {output_file}")
        evaluation.to_file(
            filepath=output_file,
            model_name=model_name,
            model_configuration=configuration.model_dump() if configuration else {},
            model_version=template.model_template_config.version or "unknown",
            model_info=model_info,
        )

        logger.info(f"Evaluation complete. Results saved to {output_file}")

        if plot:
            from chap_core.assessment.backtest_plots import create_plot_from_evaluation

            plot_path = output_file.with_suffix(".html")
            logger.info(f"Generating evaluation plot to {plot_path}")
            chart = create_plot_from_evaluation("evaluation_plot", evaluation)
            chart.save(str(plot_path))
            logger.info(f"Plot saved to {plot_path}")


def register_commands(app):
    """Register evaluate commands with the CLI app."""
    app.command(name="eval")(eval_cmd)
