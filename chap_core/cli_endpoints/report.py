"""Report generation command for CHAP CLI."""

import logging
from pathlib import Path
from typing import Annotated

import yaml
from cyclopts import Parameter

from chap_core.api_types import RunConfig
from chap_core.cli_endpoints._common import discover_geojson, load_dataset_from_csv
from chap_core.database.model_templates_and_config_tables import ModelConfiguration
from chap_core.log_config import initialize_logging
from chap_core.models.model_template import ModelTemplate

logger = logging.getLogger(__name__)


def report(
    model_path: Annotated[Path, Parameter(help="Path to an MLProject directory or GitHub URL")],
    model_artifact: Annotated[Path, Parameter(help="Path to the pre-trained model artifact")],
    dataset_csv: Annotated[Path, Parameter(help="Path to CSV file with historic data")],
    out_file: Annotated[Path, Parameter(help="Output path for the generated PDF report")],
    run_config: Annotated[RunConfig, Parameter(help="Model execution configuration")] = RunConfig(),
    model_configuration_yaml: Annotated[
        Path | None,
        Parameter(help="Path to YAML file with model configuration"),
    ] = None,
):
    """Generate a PDF report from a trained MLProject model via its ``report`` entry point."""
    initialize_logging(run_config.debug, run_config.log_file)

    geojson_path = discover_geojson(dataset_csv)
    dataset = load_dataset_from_csv(dataset_csv, geojson_path)

    configuration = None
    if model_configuration_yaml is not None:
        logger.info(f"Loading model configuration from {model_configuration_yaml}")
        configuration = ModelConfiguration.model_validate(yaml.safe_load(open(model_configuration_yaml)))

    logger.info(f"Loading model template from {model_path}")
    template = ModelTemplate.from_directory_or_github_url(
        model_path,
        ignore_env=run_config.ignore_environment,
        run_dir_type=run_config.run_directory_type,
    )

    with template:
        model = template.get_model(configuration)  # type: ignore[arg-type]
        estimator = model()
        logger.info(f"Generating report for model artifact {model_artifact}")
        estimator.report(dataset, out_file, model_artifact=model_artifact)

    logger.info(f"Report written to {out_file}")


def register_commands(app):
    """Register report commands with the CLI app."""
    app.command()(report)
