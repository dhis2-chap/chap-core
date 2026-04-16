"""Evaluation commands for CHAP CLI."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

import pandas as pd
import yaml
from cyclopts import Parameter

from chap_core import get_temp_dir
from chap_core.api_types import BackTestParams, EstimatorMode, EstimatorOptions, RunConfig
from chap_core.assessment.evaluation import Evaluation
from chap_core.assessment.prediction_evaluator import evaluate_model
from chap_core.cli_endpoints._common import (
    discover_geojson,
    get_estimator,
    get_hpo_estimator,
    load_dataset_from_csv,
    resolve_csv_path,
)
from chap_core.database.model_templates_and_config_tables import ModelConfiguration
from chap_core.datatypes import FullData
from chap_core.exceptions import NoPredictionsError
from chap_core.external.ExtendedPredictor import ExtendedPredictor
from chap_core.file_io.example_data_set import DataSetType, datasets
from chap_core.geometry import Polygons
from chap_core.hpo.base import load_search_space_from_config
from chap_core.hpo.hpoModel import Direction, HpoModel
from chap_core.hpo.objective import Objective
from chap_core.hpo.searcher import RandomSearcher
from chap_core.log_config import initialize_logging
from chap_core.models.model_template import ModelTemplate
from chap_core.models.utils import CHAP_RUNS_DIR
from chap_core.predictor import ModelType
from chap_core.spatio_temporal_data.multi_country_dataset import MultiCountryDataSet
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

if TYPE_CHECKING:
    from chap_core.models.external_model import ExternalModel

logger = logging.getLogger(__name__)


def evaluate_hpo(
    model_name: ModelType | str,
    dataset_name: DataSetType | None = None,
    dataset_country: str | None = None,
    dataset_csv: Path | None = None,
    polygons_json: Path | None = None,
    polygons_id_field: str | None = "id",
    prediction_length: int = 3,
    n_splits: int = 7,
    report_filename: str | None = str(get_temp_dir() / "report.pdf"),
    ignore_environment: bool = False,
    debug: bool = False,
    log_file: str | None = None,
    run_directory_type: Literal["latest", "timestamp", "use_existing"] | None = "timestamp",
    model_configuration_yaml: str | None = None,
    metric: str | None = "MSE",
    direction: Direction = "minimize",
    do_hpo: bool | None = True,
):
    """
    Same as evaluate, but has three added arguments and a if check on argument do_hpo.
    """
    initialize_logging(debug, log_file)
    if dataset_name is None:
        assert dataset_csv is not None, "Must specify a dataset name or a dataset csv file"
        logging.info(f"Loading dataset from {dataset_csv}")
        dataset = DataSet.from_csv(dataset_csv, FullData)
        if polygons_json is not None:
            logging.info(f"Loading polygons from {polygons_json}")
            polygons = Polygons.from_file(polygons_json, id_property=polygons_id_field)
            polygons.filter_locations(dataset.locations())
            dataset.set_polygons(polygons.data)
    else:
        logger.info(f"Evaluating model {model_name} on dataset {dataset_name}")

        example_ds = datasets[dataset_name]
        dataset = example_ds.load()

        if isinstance(dataset, MultiCountryDataSet):
            assert dataset_country is not None, "Must specify a country for multi country datasets"
            assert dataset_country in dataset.countries, (
                f"Country {dataset_country} not found in dataset. Countries: {dataset.countries}"
            )
            dataset = dataset[dataset_country]  # type: ignore[assignment]

    model_configuration_yaml_list: list[str | None]
    if "," in model_name:
        model_list = model_name.split(",")
        model_configuration_yaml_list = [None for _ in model_list]
        if model_configuration_yaml is not None:
            model_configuration_yaml_list = list(model_configuration_yaml.split(","))
            assert len(model_list) == len(model_configuration_yaml_list), (
                "Number of model configurations does not match number of models"
            )
    else:
        model_list = [model_name]
        model_configuration_yaml_list = [model_configuration_yaml]

    logging.info(f"Model configuration: {model_configuration_yaml_list}")

    results_dict = {}
    for name, configuration in zip(model_list, model_configuration_yaml_list, strict=False):
        template = ModelTemplate.from_directory_or_github_url(
            name,
            base_working_dir=CHAP_RUNS_DIR,
            ignore_env=ignore_environment,
            run_dir_type=run_directory_type,
        )
        logging.info(f"Model template loaded: {template}")
        if not do_hpo:
            model_config: ModelConfiguration | None = None
            if configuration is not None:
                logger.info(f"Loading model configuration from yaml file {configuration}")
                model_config = ModelConfiguration.model_validate(yaml.safe_load(open(configuration)))
                logger.info(f"Loaded model configuration from yaml file: {model_config}")

            model = template.get_model(model_config)  # type: ignore[arg-type]
            model = model()
        else:
            if configuration is not None:
                logger.info(f"Loading model configuration from yaml file {configuration}")
                with open(configuration, encoding="utf-8") as f:
                    configs = yaml.safe_load(f)
                if not isinstance(configs, dict) or not configs:
                    raise ValueError("YAML must define a non-empty mapping of parameters")
                logger.info(f"Loaded model base configurations from yaml file: {configs}")
            else:
                configs = template.model_template_config.hpo_search_space

            configs = load_search_space_from_config(configs)
            assert metric is not None, "metric must be specified for HPO"
            # objective = Objective(template, metric, prediction_length, n_splits)
            # now with the new Objective signature, with BackTestParams
            backtest_params = BackTestParams(n_periods=prediction_length, n_splits=n_splits)
            objective = Objective(template, backtest_params, metric)
            model = HpoModel(RandomSearcher(2), objective, direction, configs)

        model_info = model.model_information
        if model_info.min_prediction_length is None or model_info.max_prediction_length is None:
            logger.warning("Model has not specified minimum and maximum predicted length")
        else:
            if model_info.min_prediction_length > prediction_length:
                raise ValueError(
                    f"The desired prediction length of {prediction_length} is less than the model's minimum prediction length of {model_info.min_prediction_length}"
                )
            elif model_info.max_prediction_length < prediction_length:
                logger.warning(
                    f"Wrapping model to extend prediction length from {model_info.max_prediction_length} to {prediction_length}. This is done iteratively, and may worsen model performance"
                )
                model = ExtendedPredictor(model, prediction_length)

        try:
            results = evaluate_model(
                estimator=model,
                data=dataset,
                prediction_length=prediction_length,
                n_test_sets=n_splits,
                report_filename=report_filename,
            )
        except NoPredictionsError as e:
            logger.error(f"No predictions were made: {e}")
            return
        print(f"Results: {results}")
        results_dict[name] = results

    data = []
    first_model = True
    for key, value in results_dict.items():
        aggregate_metric_dist = value[0]
        row = [key, *aggregate_metric_dist.values()]
        if first_model:
            data.append(["Model", *list(aggregate_metric_dist.keys())])
            first_model = False
        data.append(row)
    dataframe = pd.DataFrame(data)
    assert report_filename is not None
    csvname = Path(report_filename).with_suffix(".csv")

    dataframe.to_csv(csvname, index=False, header=False)
    logger.info(f"Evaluation complete. Results saved to {csvname}")

    return results_dict


def eval_cmd(
    model_name: Annotated[
        str,
        Parameter(
            help="Model path (local directory), GitHub URL, or chapkit service URL. "
            "Examples: /path/to/model, https://github.com/org/model, http://localhost:8000"
        ),
    ],
    dataset_csv: Annotated[
        str,
        Parameter(
            help="Path or URL to CSV file containing disease data with columns: time_period, "
            "location, disease_cases, and climate covariates (rainfall, temperature, etc.)"
        ),
    ],
    output_file: Annotated[
        Path,
        Parameter(help="Path for output NetCDF file containing evaluation results (.nc extension)"),
    ],
    backtest_params: Annotated[
        BackTestParams,
        Parameter(
            help="Backtest configuration. Use --backtest-params.n-periods for forecast horizon, "
            "--backtest-params.n-splits for number of train/test splits, "
            "--backtest-params.stride for step size between splits"
        ),
    ] = BackTestParams(n_periods=3, n_splits=7, stride=1),
    run_config: Annotated[
        RunConfig,
        Parameter(
            help="Model execution configuration. Use --run-config.is-chapkit-model for chapkit models, "
            "--run-config.debug for verbose logging, --run-config.ignore-environment to skip env setup"
        ),
    ] = RunConfig(),
    model_configuration_yaml: Annotated[
        Path | None,
        Parameter(help="Path to YAML file with model-specific configuration parameters"),
    ] = None,
    historical_context_years: Annotated[
        int,
        Parameter(
            help="Years of historical data to include for plotting context. "
            "Calculated as periods based on dataset frequency (e.g., 6 years = 312 weeks or 72 months)"
        ),
    ] = 6,
    data_source_mapping: Annotated[
        Path | None,
        Parameter(
            help="Path to JSON file mapping model covariate names to CSV column names. "
            'Format: {"model_name": "csv_column"}. Example: {"rainfall": "precipitation_mm"}'
        ),
    ] = None,
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
        chap eval --model-name https://github.com/dhis2-chap/minimalist_example_r \\
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
    from chap_core.database.model_templates_and_config_tables import ConfiguredModelDB, ModelTemplateDB

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
        if model_info.min_prediction_length is None or model_info.max_prediction_length is None:
            logger.warning("Model has not specified minimum and maximum predicted length")
        else:
            if model_info.min_prediction_length > backtest_params.n_periods:
                raise ValueError(
                    f"The desired prediction length of {backtest_params.n_periods} is less than the model's minimum prediction length of {model_info.min_prediction_length}"
                )
            elif model_info.max_prediction_length < backtest_params.n_periods:
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
    app.command()(evaluate_hpo)
    app.command(name="eval")(eval_cmd)
