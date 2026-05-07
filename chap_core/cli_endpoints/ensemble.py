"""Ensemble CLI endpoints."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, cast

import pandas as pd
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
from chap_core.ensemble._legacy_wrappers import _TemplateWithConfig
from chap_core.ensemble.ensemble_model import EnsembleModel
from chap_core.log_config import initialize_logging
from chap_core.models.model_template import ModelTemplate
from chap_core.models.utils import CHAP_RUNS_DIR

if TYPE_CHECKING:
    from collections.abc import Sequence

    from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

logger = logging.getLogger(__name__)

# Public exports for star-imports and tooling.
__all__ = [
    "evaluate_ensemble",
    "register_commands",
]


def _load_dataset(
    *,
    dataset_name: str | None,
    dataset_country: str | None,
    dataset_csv: Path | None,
    polygons_json: Path | None,
    polygons_id_field: str,
    data_source_mapping: Path | None,
) -> DataSet[Any]:
    if dataset_name:
        return load_dataset(
            dataset_country=dataset_country,
            dataset_csv=None,
            dataset_name=dataset_name,
            polygons_id_field=polygons_id_field,
            polygons_json=polygons_json,
        )

    if dataset_csv is None:
        raise ValueError("Specify either --dataset-name or --dataset-csv")

    column_mapping = None
    if data_source_mapping is not None:
        with open(data_source_mapping) as f:
            column_mapping = json.load(f)
    geojson = polygons_json or discover_geojson(dataset_csv)
    return load_dataset_from_csv(dataset_csv, geojson, column_mapping)


def _compute_metrics(flat: Any, ensemble_method: str) -> tuple[str, dict[str, float | str], pd.DataFrame]:
    metrics_dict: dict[str, float | str] = {}
    forecasts_df = pd.DataFrame(cast("Any", flat.forecasts))
    for metric_id, metric_cls in available_metrics.items():
        metric = metric_cls()
        try:
            df_metric = metric.get_global_metric(flat.observations, cast("Any", forecasts_df))
            if len(df_metric) == 1:
                metrics_dict[metric_id] = float(df_metric["metric"].iloc[0])
        except (ValueError, KeyError, TypeError) as exc:
            logger.warning("Failed to compute metric %s: %s", metric_id, exc)

    model_key = f"ensemble_{ensemble_method}"
    metrics_dict["model_name"] = model_key
    metrics_dict["ensemble_method"] = ensemble_method
    return model_key, metrics_dict, forecasts_df


def _save_reports(
    report_filename: Path,
    results: dict[str, tuple[dict[str, float | str], object]],
) -> None:
    save_results(str(report_filename), results)


def _write_meta_report(report_filename: Path, model_names: list[str], weights: Sequence[float]) -> None:
    report_path = report_filename.with_name("ensemble_meta_report.csv")
    header = "Model," + ",".join(model_names) + "\n"
    row = "ensemble_meta," + ",".join(f"{float(w):.6f}" for w in weights) + "\n"
    report_path.write_text(header + row, encoding="utf-8")


def _evaluate_ensemble_core(
    *,
    base_model_names: str,
    ensemble_method: str,
    dataset_name: str | None,
    dataset_country: str | None,
    dataset_csv: Path | None,
    polygons_json: Path | None,
    polygons_id_field: str,
    report_filename: Path,
    output_file: Path | None,
    backtest_params: BackTestParams,
    run_config: RunConfig,
    model_configuration_yaml: str | None,
    random_state: int | None,
    data_source_mapping: Path | None,
    historical_context_years: int,
    model_template_id: str,
    configured_model_id: str,
    backtest_name: str,
    use_residual_bootstrap: bool,
) -> dict[str, tuple[dict[str, float | str], pd.DataFrame]]:
    initialize_logging(run_config.debug, run_config.log_file)
    logger.info("Evaluating ensemble with base models: %s", base_model_names)

    dataset: DataSet[Any] = _load_dataset(
        dataset_name=dataset_name,
        dataset_country=dataset_country,
        dataset_csv=dataset_csv,
        polygons_json=polygons_json,
        polygons_id_field=polygons_id_field,
        data_source_mapping=data_source_mapping,
    )

    if ensemble_method not in ("deterministic", "probabilistic"):
        raise ValueError(f"ensemble_method must be 'deterministic' or 'probabilistic', not {ensemble_method!r}")

    logger.info(
        "Backtest config: n_splits=%d, n_periods=%d, stride=%d",
        backtest_params.n_splits,
        backtest_params.n_periods,
        backtest_params.stride,
    )

    model_configuration_yaml_list, base_model_list = create_model_lists(
        model_configuration_yaml=model_configuration_yaml,
        model_name=base_model_names,
    )
    logger.info("Model configurations: %s", model_configuration_yaml_list)

    base_templates_with_config: list[_TemplateWithConfig] = []
    for name, cfg_yaml in zip(base_model_list, model_configuration_yaml_list, strict=False):
        logger.info("Loading base model template from %s", name)
        template = ModelTemplate.from_directory_or_github_url(
            name,
            base_working_dir=CHAP_RUNS_DIR,
            ignore_env=run_config.ignore_environment,
            run_dir_type=run_config.run_directory_type,
            is_chapkit_model=run_config.is_chapkit_model,
        )

        model_config: ModelConfiguration | None = None
        if cfg_yaml is not None:
            logger.info("Loading model configuration from yaml file %s", cfg_yaml)
            with open(cfg_yaml, encoding="utf-8") as f:
                cfg_data = yaml.safe_load(f)
            model_config = ModelConfiguration.model_validate(cfg_data)
            logger.info("Loaded model configuration for %s", name)

        base_templates_with_config.append(_TemplateWithConfig(template, model_config))

    ensemble = EnsembleModel(
        base_templates=base_templates_with_config,
        method=ensemble_method,
        inner_val_periods=12,
        target_col="disease_cases",
        n_samples=100,
        use_residual_bootstrap=use_residual_bootstrap,
        random_state=random_state,
    )

    model_db = ModelTemplateDB(id=model_template_id, name=model_template_id, version="0.1")
    configured_db = ConfiguredModelDB(
        id=configured_model_id,
        model_template_id=model_db.id,
        model_template=model_db,
        configuration={},  # Multiple base models, so no single merged config.
    )

    evaluation = Evaluation.create(
        configured_model=configured_db,
        estimator=ensemble,
        dataset=dataset,
        backtest_params=backtest_params,
        backtest_name=backtest_name,
        historical_context_years=historical_context_years,
    )

    eval_nc = output_file or report_filename.with_suffix(".nc")
    evaluation.to_file(str(eval_nc))
    logger.info("Saved ensemble NetCDF to %s", eval_nc)

    if ensemble.weights is not None:
        logger.info("Ensemble base model weights (percent): %s", ensemble.weights)
        try:
            _write_meta_report(report_filename, base_model_list, ensemble.weights.tolist())
            logger.info("Saved ensemble meta report to %s", report_filename.with_name("ensemble_meta_report.csv"))
        except OSError as exc:
            logger.warning("Failed to write ensemble meta report: %s", exc)

    flat = evaluation.to_flat()
    model_key, metrics_dict, forecasts_df = _compute_metrics(flat, ensemble_method)
    results: dict[str, tuple[dict[str, float | str], pd.DataFrame]] = {model_key: (metrics_dict, forecasts_df)}
    _save_reports(report_filename, cast("dict[str, tuple[dict[str, float | str], object]]", results))
    return results


def evaluate_ensemble(
    base_model_names: Annotated[
        str,
        Parameter(help="Comma-separated list of base models (local folders or GitHub URLs)."),
    ],
    ensemble_method: Annotated[
        str,
        Parameter(help="Ensemble method: 'deterministic' or 'probabilistic'."),
    ] = "probabilistic",
    dataset_name: Annotated[str | None, Parameter(help="Name of a built-in dataset.")] = None,
    dataset_country: Annotated[str | None, Parameter(help="Country for multi-country datasets.")] = None,
    dataset_csv: Annotated[Path | None, Parameter(help="CSV file with disease data.")] = None,
    polygons_json: Annotated[Path | None, Parameter(help="Optional GeoJSON file.")] = None,
    polygons_id_field: Annotated[str, Parameter(help="ID field in GeoJSON.")] = "id",
    report_filename: Annotated[Path, Parameter(help="Base filename for report outputs.")] = Path("ensemble_report.csv"),
    output_file: Annotated[Path | None, Parameter(help="Output NetCDF path.")] = None,
    backtest_params: Annotated[BackTestParams, Parameter(help="Backtest configuration.")] = BackTestParams(
        n_periods=3, n_splits=7, stride=1
    ),
    run_config: Annotated[RunConfig, Parameter(help="Model execution config.")] = RunConfig(),
    model_configuration_yaml: Annotated[
        str | None,
        Parameter(
            help=(
                "Optional comma-separated list of YAML files for base model configurations. "
                "Must match --base-model-names order/length."
            )
        ),
    ] = None,
    random_state: Annotated[
        int | None,
        Parameter(help="Random seed for the ensemble meta model (e.g. 42)."),
    ] = 42,
    use_residual_bootstrap: Annotated[
        bool,
        Parameter(help="Use residual bootstrap to generate samples for deterministic ensembles."),
    ] = False,
    data_source_mapping: Annotated[Path | None, Parameter(help="Optional JSON column mapping.")] = None,
    historical_context_years: Annotated[
        int,
        Parameter(help="Historical context (years)."),
    ] = 6,
):
    return _evaluate_ensemble_core(
        base_model_names=base_model_names,
        ensemble_method=ensemble_method,
        dataset_name=dataset_name,
        dataset_country=dataset_country,
        dataset_csv=dataset_csv,
        polygons_json=polygons_json,
        polygons_id_field=polygons_id_field,
        report_filename=report_filename,
        output_file=output_file,
        backtest_params=backtest_params,
        run_config=run_config,
        model_configuration_yaml=model_configuration_yaml,
        random_state=random_state,
        use_residual_bootstrap=use_residual_bootstrap,
        data_source_mapping=data_source_mapping,
        historical_context_years=historical_context_years,
        model_template_id="ensemble_model",
        configured_model_id="cli_eval_ensemble",
        backtest_name="ensemble_evaluation",
    )


def register_commands(app):
    app.command(name="evaluate-ensemble")(evaluate_ensemble)
