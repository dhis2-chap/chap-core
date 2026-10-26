"""Simulation command for CHAP CLI."""

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

    covariates, template, configuration = load_dataset_and_template(
        model_name, dataset_csv, model_configuration_yaml, run_config
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
