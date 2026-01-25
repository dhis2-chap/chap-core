"""Evaluation commands for CHAP CLI."""

import logging
from pathlib import Path
from typing import Optional

import yaml

from chap_core.api_types import RunConfig
from chap_core.explainability import lime
from chap_core.database.model_templates_and_config_tables import ModelConfiguration
from chap_core.log_config import initialize_logging
from chap_core.models.model_template import ModelTemplate

from chap_core.cli_endpoints._common import (
    discover_geojson,
    load_dataset_from_csv,
)

logger = logging.getLogger(__name__)



def explain(
    model_name: str,
    dataset_csv: Path,
    location: str,
    horizon: int,
    historical_context_years: int = 6,
    threshold: Optional[int] = None,
    model_configuration_yaml: Optional[Path] = None,
    run_config: RunConfig = RunConfig(),
):
    """
    Explain a model prediction by providing variable contribution weighting.

    This command accepts a model and a dataset, with the location and horizon
    providing the specific prediction to explain, and then uses the LIME procedure to
    find estimated variable contributions.

    Args:
        model_name: Model identifier (path or GitHub URL)
        dataset_csv: Path to CSV file with disease data
        location: String of the location name in which to explain prediction
        horizon: Int of number of time steps in the future of which to explain prediction
        historical_context_years: Years of historical data to include as lagged
                variable features in explanation
        threshold: Optional number of disease cases above which a prediction is
                counted as a positive class instance. If not supplied, will use
                average predicted disease cases instead.
        model_configuration_yaml: Optional YAML file with model configuration
        run_config: Model run environment configuration
    """
    # TODO: Fix too much printing in console when running
    logger.info(f"Evaluating model {model_name} using LIME")

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

        # TODO: In the future, should load an already trained pickle object or something
        logger.info(
            f"Training model..."
        )
        estimator.train(dataset)

        logger.info(
            f"Generating explanation for {location}, {horizon} time steps into the future."
        )
        logger.debug(f"Including {historical_context_years} years of historical data as individual lagged variables.")
        evaluation = lime.explain(
            model=estimator,
            dataset=dataset,
            location=location,
            horizon=horizon,
            granularity=historical_context_years,  # TODO: Fix
            threshold=threshold
        )

        # TODO: Plot results and save as csv or figure o.s.
        # Example run:
        # chap explain --model_name https://github.com/chap-models/chap_auto_ewars --dataset_csv example_data/v0/historic_data.csv --location Acre --horizon 3

def register_commands(app):
    """Register evaluate commands with the CLI app."""
    app.command()(explain)
