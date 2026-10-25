"""Report generation command for CHAP CLI."""

import logging
from pathlib import Path
from typing import Annotated

from cyclopts import Parameter

from chap_core.api_types import RunConfig
from chap_core.cli_endpoints._args import (
    DatasetCsvArg,
    ModelConfigYamlArg,
    ModelNameArg,
    RunConfigArg,
)
from chap_core.cli_endpoints._common import load_dataset_and_template
from chap_core.log_config import initialize_logging
from chap_core.models.builtin import BuiltinModelTemplate

logger = logging.getLogger(__name__)


def report(
    model_name: ModelNameArg,
    dataset_csv: DatasetCsvArg,
    out_file: Annotated[Path, Parameter(help="Output path for the generated PDF report")],
    run_config: RunConfigArg = RunConfig(),
    model_configuration_yaml: ModelConfigYamlArg = None,
):
    """Train an MLProject model on the supplied dataset and generate a PDF report via its ``report`` entry point."""
    initialize_logging(run_config.debug, run_config.log_file)

    dataset, template, configuration = load_dataset_and_template(
        model_name, dataset_csv, model_configuration_yaml, run_config
    )

    if isinstance(template, BuiltinModelTemplate):
        raise ValueError(f"chap report does not support built-in models such as {model_name!r}")

    with template:
        model = template.get_model(configuration)  # type: ignore[arg-type]
        estimator = model()
        logger.info("Training model before generating report")
        estimator.train(dataset)
        logger.info("Generating report")
        estimator.report(dataset, out_file)

    logger.info(f"Report written to {out_file}")


def register_commands(app):
    """Register report commands with the CLI app."""
    app.command()(report)
