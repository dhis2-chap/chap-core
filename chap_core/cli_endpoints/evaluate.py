"""Evaluation commands for CHAP CLI."""

import logging
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
import yaml

from chap_core.api_types import BackTestParams, RunConfig
from chap_core.assessment.evaluation import Evaluation
from chap_core.assessment.prediction_evaluator import evaluate_model
from chap_core.database.model_templates_and_config_tables import ModelConfiguration
from chap_core.datatypes import FullData
from chap_core.exceptions import NoPredictionsError
from chap_core.geometry import Polygons
from chap_core.hpo.base import load_search_space_from_config
from chap_core.hpo.hpoModel import HpoModel, Direction
from chap_core.hpo.objective import Objective
from chap_core.hpo.searcher import RandomSearcher
from chap_core.log_config import initialize_logging
from chap_core.models.model_template import ModelTemplate
from chap_core.predictor import ModelType
from chap_core.spatio_temporal_data.multi_country_dataset import MultiCountryDataSet
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core import get_temp_dir
from chap_core.file_io.example_data_set import datasets, DataSetType

from chap_core.cli_endpoints._common import (
    create_model_lists,
    discover_geojson,
    get_model,
    load_dataset,
    load_dataset_from_csv,
    save_results,
)

logger = logging.getLogger(__name__)


def evaluate_hpo(
    model_name: ModelType | str,
    dataset_name: Optional[DataSetType] = None,
    dataset_country: Optional[str] = None,
    dataset_csv: Optional[Path] = None,
    polygons_json: Optional[Path] = None,
    polygons_id_field: Optional[str] = "id",
    prediction_length: int = 3,
    n_splits: int = 7,
    report_filename: Optional[str] = str(get_temp_dir() / "report.pdf"),
    ignore_environment: bool = False,
    debug: bool = False,
    log_file: Optional[str] = None,
    run_directory_type: Optional[Literal["latest", "timestamp", "use_existing"]] = "timestamp",
    model_configuration_yaml: Optional[str] = None,
    metric: Optional[str] = "MSE",
    direction: Direction = "minimize",
    do_hpo: Optional[bool] = True,
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

        dataset = datasets[dataset_name]
        dataset = dataset.load()

        if isinstance(dataset, MultiCountryDataSet):
            assert dataset_country is not None, "Must specify a country for multi country datasets"
            assert dataset_country in dataset.countries, (
                f"Country {dataset_country} not found in dataset. Countries: {dataset.countries}"
            )
            dataset = dataset[dataset_country]

    if "," in model_name:
        model_list = model_name.split(",")
        model_configuration_yaml_list = [None for _ in model_list]
        if model_configuration_yaml is not None:
            model_configuration_yaml_list = model_configuration_yaml.split(",")
            assert len(model_list) == len(model_configuration_yaml_list), (
                "Number of model configurations does not match number of models"
            )
    else:
        model_list = [model_name]
        model_configuration_yaml_list = [model_configuration_yaml]

    logging.info(f"Model configuration: {model_configuration_yaml_list}")

    results_dict = {}
    for name, configuration in zip(model_list, model_configuration_yaml_list):
        template = ModelTemplate.from_directory_or_github_url(
            name,
            base_working_dir=Path("./runs/"),
            ignore_env=ignore_environment,
            run_dir_type=run_directory_type,
        )
        logging.info(f"Model template loaded: {template}")
        if not do_hpo:
            if configuration is not None:
                logger.info(f"Loading model configuration from yaml file {configuration}")
                configuration = ModelConfiguration.model_validate(yaml.safe_load(open(configuration)))
                logger.info(f"Loaded model configuration from yaml file: {configuration}")

            model = template.get_model(configuration)
            model = model()
        else:
            if configuration is not None:
                logger.info(f"Loading model configuration from yaml file {configuration}")
                with open(configuration, "r", encoding="utf-8") as f:
                    configs = yaml.safe_load(f)
                if not isinstance(configs, dict) or not configs:
                    raise ValueError("YAML must define a non-empty mapping of parameters")
                logger.info(f"Loaded model base configurations from yaml file: {configs}")
            else:
                configs = template.model_template_config.hpo_search_space

            configs = load_search_space_from_config(configs)
            objective = Objective(template, metric, prediction_length, n_splits)
            model = HpoModel(RandomSearcher(2), objective, direction, configs)
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
        row = [key]
        for k, v in aggregate_metric_dist.items():
            row.append(v)
        if first_model:
            data.append(["Model"] + list(aggregate_metric_dist.keys()))
            first_model = False
        data.append(row)
    dataframe = pd.DataFrame(data)
    csvname = Path(report_filename).with_suffix(".csv")

    dataframe.to_csv(csvname, index=False, header=False)
    logger.info(f"Evaluation complete. Results saved to {csvname}")

    return results_dict


def evaluate(
    model_name: ModelType | str,
    dataset_name: Optional[DataSetType] = None,
    dataset_country: Optional[str] = None,
    dataset_csv: Optional[Path] = None,
    polygons_json: Optional[Path] = None,
    polygons_id_field: Optional[str] = "id",
    prediction_length: int = 6,
    n_splits: int = 7,
    report_filename: Optional[str] = str(get_temp_dir() / "report.pdf"),
    ignore_environment: bool = False,
    debug: bool = False,
    log_file: Optional[str] = None,
    run_directory_type: Optional[Literal["latest", "timestamp", "use_existing"]] = "timestamp",
    model_configuration_yaml: Optional[str] = None,
    is_chapkit_model: bool = False,
):
    initialize_logging(debug, log_file)
    logger.info(f"Evaluating model {model_name}")
    dataset = load_dataset(dataset_country, dataset_csv, dataset_name, polygons_id_field, polygons_json)
    model_configuration_yaml_list, model_list = create_model_lists(model_configuration_yaml, model_name)
    logger.info(f"Model configuration: {model_configuration_yaml_list}")
    results_dict = {}
    for name, configuration in zip(model_list, model_configuration_yaml_list):
        model = get_model(configuration, ignore_environment, is_chapkit_model, name, run_directory_type)
        try:
            results = evaluate_model(
                model,
                dataset,
                prediction_length=prediction_length,
                n_test_sets=n_splits,
                report_filename=report_filename,
            )
        except NoPredictionsError as e:
            logger.error(f"No predictions were made: {e}")
            return
        results_dict[name] = results

    save_results(report_filename, results_dict)

    return results_dict


def evaluate2(
    model_name: str,
    dataset_csv: Path,
    output_file: Path,
    backtest_params: BackTestParams = BackTestParams(n_periods=3, n_splits=7, stride=1),
    run_config: RunConfig = RunConfig(),
    model_configuration_yaml: Optional[Path] = None,
    historical_context_years: int = 6,
):
    """
    Evaluate a single model and export results to NetCDF format using xarray.

    This command evaluates one model and outputs evaluation results in NetCDF format
    for easy integration with scientific tools. GeoJSON polygons are automatically
    discovered from a file with the same name as the CSV but with .geojson extension.

    Args:
        model_name: Model identifier (path or GitHub URL)
        dataset_csv: Path to CSV file with disease data
        output_file: Path to output NetCDF file
        backtest_params: Backtest configuration (n_periods, n_splits, stride)
        run_config: Model run environment configuration
        model_configuration_yaml: Optional YAML file with model configuration
        historical_context_years: Years of historical data to include for plotting
            context (default: 6). Number of periods is calculated based on dataset
            period type (e.g., 6 years = 312 weeks or 72 months).
    """
    from chap_core.database.model_templates_and_config_tables import ConfiguredModelDB, ModelTemplateDB

    logger.info(f"Evaluating model {model_name} with xarray/NetCDF output")

    initialize_logging(run_config.debug, run_config.log_file)

    geojson_path = discover_geojson(dataset_csv)
    dataset = load_dataset_from_csv(dataset_csv, geojson_path)

    configuration = None
    if model_configuration_yaml is not None:
        logger.info(f"Loading model configuration from {model_configuration_yaml}")
        configuration = ModelConfiguration.model_validate(yaml.safe_load(open(model_configuration_yaml)))

    logger.info(f"Loading model template from {model_name}")
    template = ModelTemplate.from_directory_or_github_url(
        model_name,
        base_working_dir=Path("./runs/"),
        ignore_env=run_config.ignore_environment,
        run_dir_type=run_config.run_directory_type,
        is_chapkit_model=run_config.is_chapkit_model,
    )

    with template:
        model = template.get_model(configuration)
        estimator = model()

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
        )

        logger.info(f"Evaluation complete. Results saved to {output_file}")


def register_commands(app):
    """Register evaluate commands with the CLI app."""
    app.command()(evaluate_hpo)
    app.command()(evaluate)
    app.command()(evaluate2)
