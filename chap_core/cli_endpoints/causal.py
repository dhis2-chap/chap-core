"""Causal counterfactual scenario comparison command."""

from __future__ import annotations

import logging
from pathlib import Path  # noqa: TC003 — used at runtime via cyclopts get_type_hints()
from typing import Annotated

import pandas as pd
from cyclopts import Parameter

from chap_core.api_types import RunConfig
from chap_core.cli_endpoints._common import (
    discover_geojson,
    get_estimator,
    load_dataset_from_csv,
    resolve_csv_path,
)

logger = logging.getLogger(__name__)


def _validate_datasets(original_df: pd.DataFrame, cf_df: pd.DataFrame, counterfactual_columns: list[str]) -> None:
    """Validate that two datasets are compatible for counterfactual comparison.

    Raises ValueError if any of the following conditions are not met:
    - Both datasets cover exactly the same (location, time_period) pairs.
    - Every column listed in counterfactual_columns is present in both datasets.
    - At least one value differs in each listed column between the two datasets.
    """
    original_periods = set(zip(original_df["location"], original_df["time_period"], strict=False))
    cf_periods = set(zip(cf_df["location"], cf_df["time_period"], strict=False))
    if original_periods != cf_periods:
        raise ValueError("Datasets do not cover the same time periods and locations.")

    for col in counterfactual_columns:
        if col not in original_df.columns:
            raise ValueError(f"Column '{col}' not found in original dataset.")
        if col not in cf_df.columns:
            raise ValueError(f"Column '{col}' not found in counterfactual dataset.")

    identical_cols = [col for col in counterfactual_columns if original_df[col].equals(cf_df[col])]
    if identical_cols:
        raise ValueError(f"No differences found in counterfactual columns: {identical_cols}")


def causal_cmd(
    model_name: Annotated[str, Parameter(help="Model path (local directory), GitHub URL, or chapkit service URL")],
    dataset_csv: Annotated[str, Parameter(help="Path or URL to the original dataset CSV")],
    counterfactual_csv: Annotated[str, Parameter(help="Path or URL to the counterfactual dataset CSV")],
    counterfactual_columns: Annotated[list[str], Parameter(help="Column names that hold counterfactual values")],
    split_period: Annotated[
        str,
        Parameter(help="Period string where training ends and prediction begins (e.g. '2023-01' or '2023W01')"),
    ],
    output_file: Annotated[
        Path,
        Parameter(help="Path for original predictions NetCDF file; counterfactual saved to {stem}_cf.nc"),
    ],
    run_config: Annotated[
        RunConfig,
        Parameter(help="Model execution configuration"),
    ] = RunConfig(),
    model_configuration_yaml: Annotated[
        Path | None,
        Parameter(help="Path to YAML file with model-specific configuration parameters"),
    ] = None,
):
    """Train a model on the original dataset up to split_period and predict on both datasets.

    Validates that the two datasets share the same time periods and locations, that the listed
    counterfactual columns are present in both, and that they actually differ. Then trains the
    model once on the original data up to (but not including) split_period and generates
    predictions from split_period to the end of the dataset for both the original and
    counterfactual inputs.

    Results are written to two NetCDF files: output_file (original) and
    output_file with a _cf suffix (counterfactual).

    Examples:
        # Train on original data up to 2023-01, predict on both scenarios
        chap causal --model-name https://github.com/dhis2-chap/minimalist_example_lag \\
            --dataset-csv ./data/original.csv \\
            --counterfactual-csv ./data/counterfactual.csv \\
            --counterfactual-columns rainfall \\
            --split-period 2023-01 \\
            --output-file ./results/causal.nc

    >>> import io, contextlib
    >>> from chap_core.cli import app
    >>> buf = io.StringIO()
    >>> try:
    ...     with contextlib.redirect_stdout(buf):
    ...         app(["causal", "--help"])
    ... except SystemExit:
    ...     pass
    >>> for flag in ("--model-name", "--dataset-csv", "--counterfactual-csv",
    ...              "--counterfactual-columns", "--split-period", "--output-file"):
    ...     assert flag in buf.getvalue(), f"{flag!r} missing from --help output"
    """
    from chap_core.assessment.dataset_splitting import train_test_split
    from chap_core.assessment.evaluation import Evaluation
    from chap_core.database.model_templates_and_config_tables import ConfiguredModelDB, ModelTemplateDB
    from chap_core.datatypes import SamplesWithTruth
    from chap_core.log_config import initialize_logging
    from chap_core.models.model_template import ModelTemplate
    from chap_core.models.utils import CHAP_RUNS_DIR
    from chap_core.rest_api.data_models import BacktestCreate
    from chap_core.time_period import TimePeriod

    initialize_logging(run_config.debug, run_config.log_file)

    original_csv_path, url_geojson_path = resolve_csv_path(dataset_csv)
    cf_csv_path, _ = resolve_csv_path(counterfactual_csv)
    geojson_path = url_geojson_path or discover_geojson(original_csv_path)

    original_df = pd.read_csv(original_csv_path)
    cf_df = pd.read_csv(cf_csv_path)
    _validate_datasets(original_df, cf_df, counterfactual_columns)

    original_dataset = load_dataset_from_csv(original_csv_path, geojson_path)
    cf_dataset = load_dataset_from_csv(cf_csv_path, geojson_path)

    logger.info(f"Loading model template from {model_name}")
    template = ModelTemplate.from_directory_or_github_url(
        model_name,
        base_working_dir=CHAP_RUNS_DIR,
        ignore_env=run_config.ignore_environment,
        run_dir_type=run_config.run_directory_type,
        is_chapkit_model=run_config.is_chapkit_model,
    )

    with template:
        estimator, configuration = get_estimator(template, model_configuration_yaml)
        model_info = estimator.model_information

        split_period_obj = TimePeriod.parse(split_period)

        train_data, original_test_data = train_test_split(original_dataset, split_period_obj)
        _, cf_test_data = train_test_split(cf_dataset, split_period_obj)

        logger.info(f"Training on original dataset up to {train_data.period_range[-1]}")
        predictor = estimator.train(train_data)

        logger.info("Predicting on original dataset")
        original_predictions = predictor.predict(train_data, original_test_data.remove_field("disease_cases"))

        logger.info("Predicting on counterfactual dataset")
        cf_predictions = predictor.predict(train_data, cf_test_data.remove_field("disease_cases"))

        original_swt = original_test_data.merge(original_predictions, result_dataclass=SamplesWithTruth)
        cf_swt = cf_test_data.merge(cf_predictions, result_dataclass=SamplesWithTruth)

        model_template_db = ModelTemplateDB(
            id=template.model_template_config.name,
            name=template.model_template_config.name,
            version=template.model_template_config.version or "unknown",
        )
        configured_model_db = ConfiguredModelDB(
            id="cli_causal",
            model_template_id=model_template_db.id,
            model_template=model_template_db,
            configuration=configuration.model_dump() if configuration else {},
        )
        backtest_info = BacktestCreate(
            name="causal_evaluation",
            dataset_id=0,
            model_id=configured_model_db.id,
        )
        last_train_period = train_data.period_range[-1]

        eval_original = Evaluation.from_samples_with_truth(
            [original_swt], last_train_period, configured_model_db, backtest_info
        )
        eval_cf = Evaluation.from_samples_with_truth([cf_swt], last_train_period, configured_model_db, backtest_info)

        cf_output_file = output_file.with_stem(output_file.stem + "_cf")
        shared_kwargs = {
            "model_name": model_name,
            "model_configuration": configuration.model_dump() if configuration else {},
            "model_version": template.model_template_config.version or "unknown",
            "model_info": model_info,
        }

        logger.info(f"Saving original predictions to {output_file}")
        eval_original.to_file(output_file, **shared_kwargs)

        logger.info(f"Saving counterfactual predictions to {cf_output_file}")
        eval_cf.to_file(cf_output_file, **shared_kwargs)


def register_commands(app):
    app.command(name="causal")(causal_cmd)
