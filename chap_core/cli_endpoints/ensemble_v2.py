"""Ensemble CLI (v2)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated

import yaml
from cyclopts import Parameter

from chap_core.api_types import BackTestParams, RunConfig
from chap_core.assessment.evaluation import Evaluation
from chap_core.assessment.metrics import available_metrics
from chap_core.cli_endpoints._common import (
    create_model_lists,
    discover_geojson,
    load_dataset,
    load_dataset_from_csv,
    save_results,
)
from chap_core.database.model_templates_and_config_tables import (
    ConfiguredModelDB,
    ModelConfiguration,
    ModelTemplateDB,
)
from chap_core.ensemble.ensemble_model_v2 import EnsembleModel
from chap_core.log_config import initialize_logging
from chap_core.models.model_template import ModelTemplate
from chap_core.models.utils import CHAP_RUNS_DIR

logger = logging.getLogger(__name__)


def evaluate_ensemble_v2(
    base_model_names: Annotated[str, Parameter(help="Comma-separated base models (paths/URLs).")],
    ensemble_method: Annotated[str, Parameter(help="'deterministic' or 'probabilistic'.")] = "probabilistic",
    dataset_name: Annotated[str | None, Parameter(help="Built-in dataset name.")] = None,
    dataset_country: Annotated[str | None, Parameter(help="Country for dataset.")] = None,
    dataset_csv: Annotated[Path | None, Parameter(help="CSV dataset path.")] = None,
    polygons_json: Annotated[Path | None, Parameter(help="GeoJSON file.")] = None,
    polygons_id_field: Annotated[str, Parameter(help="GeoJSON ID field.")] = "id",
    report_filename: Annotated[Path, Parameter(help="Report CSV base filename.")] = Path("ensemble_v2_report.csv"),
    output_file: Annotated[Path | None, Parameter(help="NetCDF output path.")] = None,
    backtest_params: Annotated[BackTestParams, Parameter(help="Backtest configuration.")] = BackTestParams(
        n_periods=3, n_splits=7, stride=1
    ),
    run_config: Annotated[RunConfig, Parameter(help="Run configuration.")] = RunConfig(),
    model_configuration_yaml: Annotated[Path | None, Parameter(help="Common model config YAML.")] = None,
    data_source_mapping: Annotated[Path | None, Parameter(help="JSON column mapping.")] = None,
    historical_context_years: Annotated[int, Parameter(help="Years of historical context.")] = 6,
):
    initialize_logging(run_config.debug, run_config.log_file)
    logger.info("Evaluating ensemble_v2 with base models: %s", base_model_names)

    # dataset
    if dataset_name:
        dataset = load_dataset(
            dataset_country=dataset_country,
            dataset_csv=None,
            dataset_name=dataset_name,
            polygons_id_field=polygons_id_field,
            polygons_json=polygons_json,
        )
    else:
        if dataset_csv is None:
            raise ValueError("Specify either --dataset-name or --dataset-csv")
        column_mapping = None
        if data_source_mapping is not None:
            with open(data_source_mapping) as f:
                column_mapping = json.load(f)
        geojson = polygons_json or discover_geojson(dataset_csv)
        dataset = load_dataset_from_csv(dataset_csv, geojson, column_mapping)

    # base models
    _, base_model_list = create_model_lists(model_configuration_yaml=None, model_name=base_model_names)

    configuration: ModelConfiguration | None = None
    if model_configuration_yaml is not None:
        with open(model_configuration_yaml) as f:
            configuration = ModelConfiguration.model_validate(yaml.safe_load(f))

    templates: list[ModelTemplate] = []
    for name in base_model_list:
        logger.info("Loading base model template from %s", name)
        tpl = ModelTemplate.from_directory_or_github_url(
            name,
            base_working_dir=CHAP_RUNS_DIR,
            ignore_env=run_config.ignore_environment,
            run_dir_type=run_config.run_directory_type,
            is_chapkit_model=run_config.is_chapkit_model,
        )
        templates.append(tpl)

    if ensemble_method not in ("deterministic", "probabilistic"):
        raise ValueError(f"ensemble_method must be 'deterministic' or 'probabilistic', not {ensemble_method!r}")

    logger.info(
        "Using ensemble_method=%s, backtest=%d splits x %d periods (stride=%d)",
        ensemble_method,
        backtest_params.n_splits,
        backtest_params.n_periods,
        backtest_params.stride,
    )

    ensemble = EnsembleModel(
        base_templates=templates,
        method=ensemble_method,
        inner_val_periods=12,
        target_col="disease_cases",
        n_samples=100,
        use_residual_bootstrap=(ensemble_method == "deterministic"),
    )

    model_db = ModelTemplateDB(id="ensemble_model_v2", name="ensemble_model_v2", version="0.1")
    configured_db = ConfiguredModelDB(
        id="cli_eval_ensemble_v2",
        model_template_id=model_db.id,
        model_template=model_db,
        configuration=configuration.model_dump() if configuration else {},
    )

    evaluation = Evaluation.create(
        configured_model=configured_db,
        estimator=ensemble,
        dataset=dataset,
        backtest_params=backtest_params,
        backtest_name="ensemble_evaluation_v2",
        historical_context_years=historical_context_years,
    )

    eval_nc = output_file or report_filename.with_suffix(".nc")
    evaluation.to_file(str(eval_nc))
    logger.info("Saved ensemble_v2 NetCDF to %s", eval_nc)

    if ensemble.weights is not None:
        logger.info("Ensemble_v2 base model weights (percent): %s", ensemble.weights)

    flat = evaluation.to_flat()
    metrics_dict: dict[str, float] = {}
    for metric_id, metric_cls in available_metrics.items():
        metric = metric_cls()
        try:
            df_metric = metric.get_global_metric(flat.observations, flat.forecasts)
            if len(df_metric) == 1:
                metrics_dict[metric_id] = float(df_metric["metric"].iloc[0])
        except Exception as e:
            logger.warning("Failed to compute metric %s: %s", metric_id, e)

    model_key = f"ensemble_{ensemble_method}"
    metrics_dict["model_name"] = model_key
    metrics_dict["ensemble_method"] = ensemble_method

    results = {model_key: (metrics_dict, flat.forecasts)}
    save_results(str(report_filename), results)
    logger.info("Saved ensemble_v2 results to %s (and .i.csv)", report_filename)
    return results


def register_commands_v2(app):
    app.command(name="evaluate-ensemble-v2")(evaluate_ensemble_v2)
