"""Evaluation commands for CHAP CLI."""

from __future__ import annotations

import logging
from pathlib import Path  # noqa: TC003 — used at runtime via cyclopts get_type_hints()
from typing import Annotated

from cyclopts import Parameter
from pydantic import BaseModel

from chap_core.api_types import RunConfig
from chap_core.cli_endpoints._common import (
    discover_geojson,
    load_dataset_from_csv,
)

logger = logging.getLogger(__name__)


class LimeParams(BaseModel):
    """Configuration for the LIME explainability pipeline."""

    granularity: int = 10
    num_perturbations: int = 300
    surrogate_name: str = "ridge"
    segmenter_name: str = "uniform"
    sampler_name: str = "background"
    weighter_name: str = "pairwise"
    seed: int | None = None
    timed: bool = False
    adaptive: bool = False
    last_n: int | None = None


def explain_lime(
    model_path: Annotated[
        Path,
        Parameter(help="Path to a trained model run directory"),
    ],
    dataset_csv: Annotated[
        Path,
        Parameter(help="Path to CSV file with disease data"),
    ],
    location: Annotated[
        str,
        Parameter(help="Location name for which to explain the prediction"),
    ],
    horizon: Annotated[
        int,
        Parameter(help="Number of time steps into the future to explain"),
    ] = 1,
    lime_params: Annotated[
        LimeParams,
        Parameter(
            help="LIME pipeline configuration. Use --lime-params.granularity for segment count, "
            "--lime-params.segmenter-name for segmentation strategy, "
            "--lime-params.sampler-name for perturbation strategy, "
            "--lime-params.num-perturbations for sample count, "
            "--lime-params.adaptive to enable adaptive mode"
        ),
    ] = LimeParams(),
    run_config: Annotated[
        RunConfig,
        Parameter(help="Model execution configuration"),
    ] = RunConfig(),
    model_configuration_yaml: Annotated[
        Path | None,
        Parameter(help="Path to YAML file with model configuration"),
    ] = None,
    save: Annotated[
        bool,
        Parameter(help="Save explanation as a markdown file under runs/explainability/"),
    ] = True,
):
    """
    Explain a model prediction by providing variable contribution weighting.
    """
    import yaml

    from chap_core.database.model_templates_and_config_tables import ModelConfiguration
    from chap_core.log_config import initialize_logging
    from chap_core.models.model_template import ModelTemplate

    # TODO: Fix too much printing in console when running
    logger.info(f"Evaluating model {model_path} using LIME")

    if lime_params.adaptive:
        from chap_core.explainability.lime import explain_adaptive as explain_fn
    else:
        from chap_core.explainability.lime import explain as explain_fn

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
        run_dir_type="use_existing",
        is_chapkit_model=run_config.is_chapkit_model,
    )

    with template:
        model = template.get_model(configuration)  # type: ignore[arg-type]
        estimator = model()

        logger.info(f"Generating explanation for {location}, {horizon} time steps into the future.")

        explain_fn(
            model=estimator,
            dataset=dataset,
            location=location,
            horizon=horizon,
            num_perturbations=lime_params.num_perturbations,
            surrogate_name=lime_params.surrogate_name,
            segmenter_name=lime_params.segmenter_name,
            sampler_name=lime_params.sampler_name,
            weighter_name=lime_params.weighter_name,
            seed=lime_params.seed,
            timed=lime_params.timed,
            granularity=lime_params.granularity,
            last_n=lime_params.last_n,
            save=save,
        )


def register_commands(app):
    """Register evaluate commands with the CLI app."""
    app.command()(explain_lime)
