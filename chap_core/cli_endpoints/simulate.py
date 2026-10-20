"""Simulation command for CHAP CLI."""

import logging
from pathlib import Path
from typing import Annotated

import yaml
from cyclopts import Parameter

from chap_core.api_types import RunConfig
from chap_core.cli_endpoints._args import (
    DatasetCsvArg,
    ModelConfigYamlArg,
    ModelNameArg,
    RunConfigArg,
)
from chap_core.cli_endpoints._common import discover_geojson, load_dataset_from_csv, resolve_csv_path
from chap_core.database.model_templates_and_config_tables import ModelConfiguration
from chap_core.log_config import initialize_logging
from chap_core.models.model_template import ModelTemplate

logger = logging.getLogger(__name__)


def simulate(
    model_name: ModelNameArg,
    dataset_csv: DatasetCsvArg,
    out_file: Annotated[Path, Parameter(help="Output CSV path for the covariates with the simulated target added")],
    run_config: RunConfigArg = RunConfig(),
    model_configuration_yaml: ModelConfigYamlArg = None,
):
    """Simulate disease data for the locations and periods in a covariate CSV via a model's ``simulate`` entry point."""
    initialize_logging(run_config.debug, run_config.log_file)

    csv_path, url_geojson_path = resolve_csv_path(dataset_csv)
    geojson_path = url_geojson_path or discover_geojson(csv_path)
    covariates = load_dataset_from_csv(csv_path, geojson_path)

    configuration = None
    if model_configuration_yaml is not None:
        logger.info(f"Loading model configuration from {model_configuration_yaml}")
        configuration = ModelConfiguration.model_validate(yaml.safe_load(open(model_configuration_yaml)))

    logger.info(f"Loading model template from {model_name}")
    template = ModelTemplate.from_directory_or_github_url(
        model_name,
        ignore_env=run_config.ignore_environment,
        run_dir_type=run_config.run_directory_type,
    )

    with template:
        model = template.get_model(configuration)  # type: ignore[arg-type]
        simulated = model().simulate(covariates)
    assert simulated is not None

    simulated.to_csv(str(out_file))
    if simulated.polygons is not None:
        from chap_core.geometry import Polygons

        Polygons(simulated.polygons).to_file(Path(out_file).with_suffix(".geojson"))
    logger.info(f"Simulated data written to {out_file}")


def register_commands(app):
    """Register simulate commands with the CLI app."""
    app.command()(simulate)
