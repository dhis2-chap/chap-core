"""Common helper functions shared across CLI commands."""

import logging
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import pooch
import yaml

from chap_core.api_types import BackTestParams
from chap_core.database.model_templates_and_config_tables import ModelConfiguration
from chap_core.file_io.example_data_set import datasets
from chap_core.geometry import Polygons
from chap_core.hpo.base import load_search_space_from_config
from chap_core.hpo.hpoModel import HpoModel
from chap_core.hpo.objective import Objective
from chap_core.hpo.searcher import RandomSearcher, Searcher
from chap_core.models.external_model import ExternalModel
from chap_core.models.model_template import ModelTemplate
from chap_core.models.utils import CHAP_RUNS_DIR
from chap_core.spatio_temporal_data.multi_country_dataset import MultiCountryDataSet
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet, DataSetMetaData

logger = logging.getLogger(__name__)


def append_to_csv(file_object, data_frame: pd.DataFrame):
    data_frame.to_csv(file_object, mode="a", header=False)


def get_model(
    configuration: str | None,
    ignore_environment: bool,
    is_chapkit_model: bool,
    name: str,
    run_directory_type: Literal["latest", "timestamp", "use_existing"] | None,
) -> Any:
    template = ModelTemplate.from_directory_or_github_url(
        name,
        base_working_dir=CHAP_RUNS_DIR,
        ignore_env=ignore_environment,
        run_dir_type=run_directory_type,
        is_chapkit_model=is_chapkit_model,
    )
    logger.info(f"Model template loaded: {template}")
    model_config: ModelConfiguration | None = None
    if configuration is not None:
        logger.info(f"Loading model configuration from yaml file {configuration}")
        model_config = ModelConfiguration.model_validate(yaml.safe_load(open(configuration)))
        logger.info(f"Loaded model configuration from yaml file: {model_config}")

    model = template.get_model(model_config)  # type: ignore[arg-type]
    model = model()
    return model


def save_results(report_filename: str, results_dict: dict[Any, Any]) -> None:
    data: list[list[Any]] = []
    full_data: dict[Any, pd.DataFrame] = {}
    first_model = True
    for key, value in results_dict.items():
        aggregate_metric_dist = value[0]
        row: list[Any] = [key, *aggregate_metric_dist.values()]
        if first_model:
            data.append(["Model", *list(aggregate_metric_dist.keys())])
            first_model = False
        data.append(row)
        full_data[key] = pd.DataFrame(value[1])

    dataframe = pd.DataFrame(data)
    csvname = Path(report_filename).with_suffix(".csv")
    for i, (model_name, results) in enumerate(full_data.items()):
        csvname_full = Path(report_filename).with_suffix(f".{i}.csv")
        results.to_csv(csvname_full, index=False)
        logger.info(f"Wrote detailed results for {model_name} to {csvname_full}")

    dataframe.to_csv(csvname, index=False, header=False)
    logger.info(f"Evaluation complete. Results saved to {csvname}")


def create_model_lists(model_configuration_yaml: str | None, model_name: str) -> tuple[list[str | None], list[str]]:
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
    return model_configuration_yaml_list, model_list


def resolve_csv_path(dataset_csv: str | Path) -> tuple[Path, Path | None]:
    """If dataset_csv is a URL, download it and return local path + optional geojson path.

    For local paths, returns the path unchanged with no geojson path.
    For URLs, downloads the CSV using pooch and also attempts to download
    a companion .geojson file from the same URL with the extension replaced.
    """
    dataset_csv = str(dataset_csv)
    if not dataset_csv.startswith(("http://", "https://")):
        return Path(dataset_csv), None

    logger.info(f"Downloading CSV from URL: {dataset_csv}")
    local_path = Path(pooch.retrieve(dataset_csv, known_hash=None))

    geojson_url = dataset_csv.replace(".csv", ".geojson")
    geojson_path = None
    try:
        geojson_path = Path(pooch.retrieve(geojson_url, known_hash=None))
        logger.info(f"Downloaded companion GeoJSON from: {geojson_url}")
    except Exception:
        logger.debug(f"No companion GeoJSON found at: {geojson_url}")

    return local_path, geojson_path


def discover_geojson(csv_path: Path) -> Path | None:
    """
    Discover GeoJSON file alongside CSV file.

    Looks for a file with the same name as the CSV but with .geojson extension.

    Args:
        csv_path: Path to CSV file

    Returns:
        Path to GeoJSON file if it exists, None otherwise
    """
    geojson_path = Path(str(csv_path).replace(".csv", ".geojson"))
    if geojson_path.exists():
        return geojson_path
    return None


def load_dataset_from_csv(
    csv_path: Path,
    geojson_path: Path | None = None,
    column_mapping: dict[str, str] | None = None,
) -> DataSet:
    """
    Load dataset from CSV file with optional GeoJSON polygons.

    Args:
        csv_path: Path to CSV file with disease data
        geojson_path: Optional path to GeoJSON file with polygon boundaries
        column_mapping: Optional mapping from covariate names (keys) to CSV column names (values).
            If provided, columns will be renamed before creating the DataSet.

    Returns:
        DataSet loaded from CSV with polygons if provided
    """
    logging.info(f"Loading dataset from {csv_path}")

    dataset: DataSet
    if column_mapping is not None:
        logging.info(f"Applying column mapping: {column_mapping}")
        df = pd.read_csv(csv_path)
        # Rename columns: mapping is {target_name: source_name}, so swap for rename
        df.rename(columns={v: k for k, v in column_mapping.items()}, inplace=True)
        dataset = DataSet.from_pandas(df)
        dataset.metadata = DataSetMetaData(name=str(Path(csv_path).stem), filename=str(csv_path))
    else:
        dataset = DataSet.from_csv(csv_path)

    if geojson_path is not None:
        logging.info(f"Loading polygons from {geojson_path}")
        polygons = Polygons.from_file(geojson_path, id_property="id")
        polygons.filter_locations(dataset.locations())
        dataset.set_polygons(polygons.data)

    return dataset


def load_dataset(
    dataset_country: str | None,
    dataset_csv: Path | None,
    dataset_name: Any | None,
    polygons_id_field: str | None,
    polygons_json: Path | None,
) -> DataSet:
    dataset: DataSet
    if dataset_name is None:
        assert dataset_csv is not None, "Must specify a dataset name or a dataset csv file"
        logging.info(f"Loading dataset from {dataset_csv}")
        dataset = DataSet.from_csv(dataset_csv)
        if polygons_json is not None:
            logging.info(f"Loading polygons from {polygons_json}")
            polygons = Polygons.from_file(polygons_json, id_property=polygons_id_field)
            polygons.filter_locations(dataset.locations())
            dataset.set_polygons(polygons.data)
    else:
        logger.info(f"Evaluating model on dataset {dataset_name}")
        example_ds = datasets[dataset_name]
        dataset = example_ds.load()

        if isinstance(dataset, MultiCountryDataSet):
            assert dataset_country is not None, "Must specify a country for multi country datasets"
            assert dataset_country in dataset.countries, (
                f"Country {dataset_country} not found in dataset. Countries: {dataset.countries}"
            )
            dataset = dataset[dataset_country]  # type: ignore[assignment]
    return dataset


def get_estimator(
    template: ModelTemplate,
    model_configuration_yaml: Path | None,
) -> tuple[ExternalModel, ModelConfiguration | None]:
    """
    Build a plain estimator from a model template and optional configuration yaml file.
    Returns both the estimator and the parsed configuration so callers can reuse the
    configuration for metadata/export.
    """
    configuration = None
    if model_configuration_yaml is not None:
        logger.info(f"Loading model configuration from {model_configuration_yaml}")
        configuration = ModelConfiguration.model_validate(yaml.safe_load(open(model_configuration_yaml)))

    model = template.get_model(configuration)  # type: ignore[arg-type]
    estimator = model()
    return estimator, configuration


def get_hpo_estimator(
    template: ModelTemplate,
    model_configuration_yaml: Path | None,
    backtest_params: BackTestParams,
    metric: str,
    searcher: Searcher | None = None,
) -> HpoModel:
    """
    Build an HPO-backend estimator from either:
    - an explicit YAML search space, or
    - the template's built-in hpo_search_space
    """
    if model_configuration_yaml is not None:
        logger.info(f"Loading model configuration from {model_configuration_yaml}")
        with open(model_configuration_yaml, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        if not isinstance(config, dict) or not config:
            raise ValueError("YAML must define a non-empty mapping of parameters")
    else:
        config = template.model_template_config.hpo_search_space

    search_space = load_search_space_from_config(config)
    objective = Objective(template, backtest_params, metric)

    # Seacher object can be passed as argument: searcher: Searcher | None = None,
    # return HpoModel(searcher or RandonSearcher(2) ...)
    if searcher is None:
        searcher = RandomSearcher(3)
    return HpoModel(searcher, objective, "minimize", search_space)
