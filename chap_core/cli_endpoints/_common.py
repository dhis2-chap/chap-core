"""Common helper functions shared across CLI commands."""

import logging
from pathlib import Path
from typing import Any, Literal, Optional

import pandas as pd
import yaml

from chap_core.database.model_templates_and_config_tables import ModelConfiguration
from chap_core.datatypes import FullData
from chap_core.geometry import Polygons
from chap_core.models.model_template import ModelTemplate
from chap_core.spatio_temporal_data.multi_country_dataset import MultiCountryDataSet
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.file_io.example_data_set import datasets

logger = logging.getLogger(__name__)


def append_to_csv(file_object, data_frame: pd.DataFrame):
    data_frame.to_csv(file_object, mode="a", header=False)


def get_model(
    configuration: str,
    ignore_environment: bool,
    is_chapkit_model: bool,
    name,
    run_directory_type: Literal["latest", "timestamp", "use_existing"] | None,
) -> Any:
    template = ModelTemplate.from_directory_or_github_url(
        name,
        base_working_dir=Path("./runs/"),
        ignore_env=ignore_environment,
        run_dir_type=run_directory_type,
        is_chapkit_model=is_chapkit_model,
    )
    logger.info(f"Model template loaded: {template}")
    if configuration is not None:
        logger.info(f"Loading model configuration from yaml file {configuration}")
        configuration = ModelConfiguration.model_validate(yaml.safe_load(open(configuration)))
        logger.info(f"Loaded model configuration from yaml file: {configuration}")

    model = template.get_model(configuration)
    model = model()
    return model


def save_results(report_filename: str | None, results_dict: dict[Any, Any]):
    data = []
    full_data = {}
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
        full_data[key] = pd.DataFrame(value[1])

    dataframe = pd.DataFrame(data)
    csvname = Path(report_filename).with_suffix(".csv")
    for i, (model_name, results) in enumerate(full_data.items()):
        csvname_full = Path(report_filename).with_suffix(f".{i}.csv")
        results.to_csv(csvname_full, index=False)
        logger.info(f"Wrote detailed results for {model_name} to {csvname_full}")

    dataframe.to_csv(csvname, index=False, header=False)
    logger.info(f"Evaluation complete. Results saved to {csvname}")


def create_model_lists(model_configuration_yaml: str | None, model_name) -> tuple[list[str], Any]:
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
    return model_configuration_yaml_list, model_list


def discover_geojson(csv_path: Path) -> Optional[Path]:
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


def load_dataset_from_csv(csv_path: Path, geojson_path: Optional[Path] = None) -> DataSet:
    """
    Load dataset from CSV file with optional GeoJSON polygons.

    Args:
        csv_path: Path to CSV file with disease data
        geojson_path: Optional path to GeoJSON file with polygon boundaries

    Returns:
        DataSet loaded from CSV with polygons if provided
    """
    logging.info(f"Loading dataset from {csv_path}")
    dataset = DataSet.from_csv(csv_path, FullData)

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
        logger.info(f"Evaluating model on dataset {dataset_name}")
        dataset = datasets[dataset_name]
        dataset = dataset.load()

        if isinstance(dataset, MultiCountryDataSet):
            assert dataset_country is not None, "Must specify a country for multi country datasets"
            assert dataset_country in dataset.countries, (
                f"Country {dataset_country} not found in dataset. Countries: {dataset.countries}"
            )
            dataset = dataset[dataset_country]
    return dataset
