"""Evaluation commands for CHAP CLI."""

from __future__ import annotations

import logging
from typing import Annotated

from cyclopts import Parameter
from pydantic import BaseModel

from chap_core.api_types import RunConfig
from chap_core.cli_endpoints._args import (  # noqa: TC001 — used at runtime via cyclopts get_type_hints()
    DatasetCsvArg,
    ModelConfigYamlArg,
    ModelNameArg,
    RunConfigArg,
)
from chap_core.cli_endpoints._common import (
    discover_geojson,
    load_dataset_from_csv,
    resolve_csv_path,
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
    with_metrics: bool = False


def _resolve_locations(requested: list[str] | None, all_locations: bool, available: list[str]) -> list[str]:
    """Resolve and validate the locations to explain.

    Exactly one of ``requested`` / ``all_locations`` must select locations:
    ``--all-locations`` returns every available location; otherwise the
    requested names are validated against ``available`` (order preserved).
    Raises ``ValueError`` with a helpful message on a missing, unknown, or
    conflicting selection.
    """
    if all_locations:
        if requested:
            raise ValueError("Pass either --location or --all-locations, not both.")
        return list(available)
    if not requested:
        raise ValueError("No location given. Pass --location <name> (repeatable) or --all-locations.")
    unknown = [loc for loc in requested if loc not in available]
    if unknown:
        raise ValueError(f"Unknown location(s): {unknown}. Valid locations: {sorted(available)}")
    return list(requested)


def explain_lime(
    model_name: ModelNameArg,
    dataset_csv: DatasetCsvArg,
    location: Annotated[
        list[str] | None,
        Parameter(
            help="Location name(s) to explain (repeatable, e.g. --location A --location B). Required unless --all-locations is set."
        ),
    ] = None,
    all_locations: Annotated[
        bool,
        Parameter(
            help="Explain every org-unit in the dataset. Warning: this runs a full perturbation loop "
            "(~num-perturbations model calls) per location, so it can be very slow on many-location datasets."
        ),
    ] = False,
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
    run_config: RunConfigArg = RunConfig(),
    model_configuration_yaml: ModelConfigYamlArg = None,
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
    logger.info(f"Evaluating model {model_name} using LIME")

    if lime_params.adaptive:
        from chap_core.explainability.lime import explain_adaptive as explain_fn
    else:
        from chap_core.explainability.lime import explain as explain_fn

    initialize_logging(run_config.debug, run_config.log_file)

    csv_path, url_geojson_path = resolve_csv_path(dataset_csv)
    geojson_path = url_geojson_path or discover_geojson(csv_path)
    dataset = load_dataset_from_csv(csv_path, geojson_path)

    # Fail fast on a bad/missing location before the (expensive) model load.
    locations = _resolve_locations(location, all_locations, list(dataset.locations()))

    configuration = None
    if model_configuration_yaml is not None:
        logger.info(f"Loading model configuration from {model_configuration_yaml}")
        with open(model_configuration_yaml) as f:
            configuration = ModelConfiguration.model_validate(yaml.safe_load(f))

    logger.info(f"Loading model template from {model_name}")
    template = ModelTemplate.from_directory_or_github_url(
        model_name,
        ignore_env=run_config.ignore_environment,
        run_dir_type="use_existing",
        is_chapkit_model=run_config.is_chapkit_model,
    )

    with template:
        model = template.get_model(configuration)  # type: ignore[arg-type]
        estimator = model()

        for loc in locations:
            logger.info(f"Generating explanation for {loc}, {horizon} time steps into the future.")

            result = explain_fn(
                model=estimator,
                dataset=dataset,
                location=loc,
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
                return_metrics=lime_params.with_metrics,
            )

            if lime_params.with_metrics:
                assert isinstance(result, tuple)  # return_metrics=True yields (results, metrics)
                _, metrics = result
                logger.info(f"Faithfulness metrics for {loc}:")
                for key, value in metrics.items():
                    logger.info(f"  {key:>15} = {value:+.4f}")


def register_commands(app):
    """Register evaluate commands with the CLI app."""
    app.command()(explain_lime)
