"""Ensemble CLI endpoints.

EXPERIMENTAL: the evaluate-ensemble command and the ensemble API behind it are
under active development and may change or be removed without notice.
"""

from __future__ import annotations

import json
import logging
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, cast

import yaml
from cyclopts import Parameter

from chap_core.api_types import BacktestParams, RunConfig
from chap_core.cli_endpoints._common import (
    create_model_lists,
    discover_geojson,
    load_dataset,
    load_dataset_from_csv,
    resolve_csv_path,
    save_results,
    warn_unused_covariates,
)
from chap_core.log_config import initialize_logging

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd

    from chap_core.database.model_templates_and_config_tables import ModelConfiguration
    from chap_core.ensemble.wrappers import TemplateWithConfig
    from chap_core.models.model_template import ModelTemplate
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
    dataset_csv: str | Path | None,
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

    # dataset_csv may be a local path or a URL; resolve_csv_path downloads the CSV
    # and any companion .geojson, matching how `chap evaluate` loads its dataset.
    csv_path, url_geojson_path = resolve_csv_path(dataset_csv)
    geojson = polygons_json or url_geojson_path or discover_geojson(csv_path)
    return load_dataset_from_csv(csv_path, geojson, column_mapping)


def _compute_metrics(flat: Any, ensemble_method: str) -> tuple[str, dict[str, float | str], pd.DataFrame]:
    import pandas as pd

    from chap_core.assessment.metrics import available_metrics

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


def _write_meta_report(
    report_filename: Path,
    model_names: list[str],
    fit_history: Sequence[tuple[Sequence[float], Sequence[float]]],
) -> Path:
    """Write one weight_percent and one coefficient row per meta-model fit.

    The path follows the report stem: a fixed filename would let two runs in the same
    directory with different ``--report-filename`` clobber each other's weights while
    their other outputs stayed distinct.

    The deterministic meta-model applies the raw coefficients, whose sum need not be 1,
    so reporting only the normalised shares would hide the scaling. Backtests with
    ``n_retrain > 1`` fit the meta-model more than once, and every round is recorded
    rather than only the last.
    """
    report_path = report_filename.with_name(f"{report_filename.stem}_meta.csv")
    lines = ["Model,round,quantity," + ",".join(model_names)]
    for round_no, (weights, coefficients) in enumerate(fit_history, start=1):
        for quantity, values in (("weight_percent", weights), ("coefficient", coefficients)):
            lines.append(f"ensemble_meta,{round_no},{quantity}," + ",".join(f"{float(v):.6f}" for v in values))
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def _prepare_base_template(
    template: ModelTemplate,
    model_config: ModelConfiguration | None,
    dataset: DataSet[Any],
    n_periods: int,
) -> TemplateWithConfig:
    """Apply the per-model checks ``chap evaluate`` applies, then bind template to config.

    Without these a base model capped at fewer periods than the backtest horizon is
    silently asked for too many, and its short prediction frame either drops rows from
    the weight fit or fails the backtest mid-run.
    """
    from chap_core.ensemble.wrappers import TemplateWithConfig

    config = template.model_template_config
    warn_unused_covariates(dataset, config, model_config)

    min_length = config.min_prediction_periods
    max_length = config.max_prediction_periods
    if min_length is None and max_length is None:
        logger.warning("Base model %s has not specified minimum and maximum predicted length", config.name)
    if min_length is not None and min_length > n_periods:
        raise ValueError(
            f"The desired prediction length of {n_periods} is less than the minimum prediction length of "
            f"{min_length} for base model {config.name}"
        )
    extend_to = None
    if max_length is not None and max_length < n_periods:
        logger.warning(
            "Wrapping base model %s to extend prediction length from %d to %d. This is done iteratively, "
            "and may worsen model performance",
            config.name,
            max_length,
            n_periods,
        )
        extend_to = n_periods
    return TemplateWithConfig(template, model_config, extend_to_prediction_length=extend_to)


def _evaluate_ensemble_core(
    *,
    base_model_names: str,
    ensemble_method: str,
    dataset_name: str | None,
    dataset_country: str | None,
    dataset_csv: str | Path | None,
    polygons_json: Path | None,
    polygons_id_field: str,
    report_filename: Path,
    output_file: Path | None,
    backtest_params: BacktestParams,
    run_config: RunConfig,
    model_configuration_yaml: str | None,
    inner_val_periods: int,
    n_samples: int,
    data_source_mapping: Path | None,
    historical_context_years: int,
    model_template_id: str,
    configured_model_id: str,
    backtest_name: str,
) -> dict[str, tuple[dict[str, float | str], pd.DataFrame]]:
    initialize_logging(run_config.debug, run_config.log_file)
    # Validated before the dataset is loaded: loading may download a remote CSV plus a
    # companion GeoJSON, which is a long wait to pay for a typo in --ensemble-method.
    if ensemble_method not in ("deterministic", "probabilistic"):
        raise ValueError(f"ensemble_method must be 'deterministic' or 'probabilistic', not {ensemble_method!r}")
    if n_samples < 1:
        raise ValueError(f"n_samples must be at least 1, got {n_samples}")
    logger.info("Evaluating ensemble with base models: %s", base_model_names)

    dataset: DataSet[Any] = _load_dataset(
        dataset_name=dataset_name,
        dataset_country=dataset_country,
        dataset_csv=dataset_csv,
        polygons_json=polygons_json,
        polygons_id_field=polygons_id_field,
        data_source_mapping=data_source_mapping,
    )

    # Imported lazily, as in evaluate.py, to keep the REST stack off the CLI startup path.
    from chap_core.rest_api.db_worker_functions import validate_and_filter_dataset_for_evaluation

    dataset = validate_and_filter_dataset_for_evaluation(
        dataset,
        target_name="disease_cases",
        n_periods=backtest_params.n_periods,
        n_splits=backtest_params.n_splits,
        stride=backtest_params.stride,
    )

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

    # Imported here rather than at module level, as in evaluate.py: these pull in
    # scipy, chapkit and the DB layer, which would otherwise land on the startup
    # path of every chap CLI command.
    from chap_core.assessment.evaluation import Evaluation
    from chap_core.database.model_templates_and_config_tables import (
        ConfiguredModelDB,
        ModelConfiguration,
        ModelTemplateDB,
    )
    from chap_core.ensemble.ensemble_model import EnsembleModel
    from chap_core.models.model_template import ModelTemplate
    from chap_core.models.utils import CHAP_RUNS_DIR

    # Templates must stay open for the whole run: for chapkit models __enter__ is what
    # starts the backing service and sets up the client, and __exit__ shuts it down.
    with ExitStack() as stack:
        base_templates_with_config: list[TemplateWithConfig] = []
        for name, cfg_yaml in zip(base_model_list, model_configuration_yaml_list, strict=False):
            logger.info("Loading base model template from %s", name)
            template = ModelTemplate.from_directory_or_github_url(
                name,
                base_working_dir=CHAP_RUNS_DIR,
                ignore_env=run_config.ignore_environment,
                run_dir_type=run_config.run_directory_type,
                is_chapkit_model=run_config.is_chapkit_model,
            )
            stack.enter_context(template)

            model_config: ModelConfiguration | None = None
            if cfg_yaml is not None:
                logger.info("Loading model configuration from yaml file %s", cfg_yaml)
                with open(cfg_yaml, encoding="utf-8") as f:
                    cfg_data = yaml.safe_load(f)
                model_config = ModelConfiguration.model_validate(cfg_data)
                logger.info("Loaded model configuration for %s", name)

            base_templates_with_config.append(
                _prepare_base_template(template, model_config, dataset, backtest_params.n_periods)
            )

        ensemble = EnsembleModel(
            base_templates=base_templates_with_config,
            method=ensemble_method,
            inner_val_periods=inner_val_periods,
            horizon=backtest_params.n_periods,
            target_col="disease_cases",
            n_samples=n_samples,
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

    if ensemble.fit_history:
        logger.info("Ensemble base model weights (percent): %s", ensemble.fit_history[-1][0])
        history = [(w.tolist(), c.tolist()) for w, c in ensemble.fit_history]
        try:
            meta_path = _write_meta_report(report_filename, base_model_list, history)
            logger.info("Saved ensemble meta report to %s", meta_path)
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
    dataset_csv: Annotated[str | None, Parameter(help="CSV file with disease data.")] = None,
    polygons_json: Annotated[Path | None, Parameter(help="Optional GeoJSON file.")] = None,
    polygons_id_field: Annotated[str, Parameter(help="ID field in GeoJSON.")] = "id",
    report_filename: Annotated[Path, Parameter(help="Base filename for report outputs.")] = Path("ensemble_report.csv"),
    output_file: Annotated[Path | None, Parameter(help="Output NetCDF path.")] = None,
    backtest_params: Annotated[BacktestParams, Parameter(help="Backtest configuration.")] = BacktestParams(
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
    inner_val_periods: Annotated[
        int,
        Parameter(
            help=(
                "Number of trailing training periods held out to fit the ensemble weights. "
                "Split into windows of --backtest-params.n-periods so base models are ranked "
                "at the horizon they are actually used at."
            )
        ),
    ] = 12,
    n_samples: Annotated[
        int,
        Parameter(
            help=(
                "Number of samples the ensemble forecast is represented by. Base models "
                "producing fewer samples are resampled up to this number, which cannot add "
                "distributional resolution they do not have."
            )
        ),
    ] = 100,
    data_source_mapping: Annotated[Path | None, Parameter(help="Optional JSON column mapping.")] = None,
    historical_context_years: Annotated[
        int,
        Parameter(help="Historical context (years)."),
    ] = 6,
):
    """EXPERIMENTAL: evaluate a stacking ensemble of several base models.

    This command is experimental. Its interface, defaults and outputs may change
    or be removed in any release, and the results should not yet be relied on for
    production evaluations.

    Trains the base models on an inner split of the training data, fits ensemble
    weights on the held-out windows, then backtests the combined model. Base models
    are given as a comma-separated list of local folders or GitHub URLs, optionally
    paired with a matching list of configuration YAML files.
    """
    logger.warning("evaluate-ensemble is EXPERIMENTAL: its interface and results may change without notice")
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
        inner_val_periods=inner_val_periods,
        n_samples=n_samples,
        data_source_mapping=data_source_mapping,
        historical_context_years=historical_context_years,
        model_template_id="ensemble_model",
        configured_model_id="cli_eval_ensemble",
        backtest_name="ensemble_evaluation",
    )


def register_commands(app):
    app.command(name="evaluate-ensemble")(evaluate_ensemble)
