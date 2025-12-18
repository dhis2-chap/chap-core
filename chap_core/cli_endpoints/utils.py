"""Utility commands for CHAP CLI."""

import dataclasses
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from chap_core.assessment.dataset_splitting import train_test_generator
from chap_core.database.model_templates_and_config_tables import ModelConfiguration
from chap_core.datatypes import FullData
from chap_core.log_config import initialize_logging
from chap_core.models.utils import get_model_template_from_directory_or_github_url
from chap_core.plotting.dataset_plot import StandardizedFeaturePlot
from chap_core.plotting.season_plot import SeasonCorrelationBarPlot
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.file_io.example_data_set import datasets

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class AreaPolygons: ...


def sanity_check_model(
    model_url: str, use_local_environement: bool = False, dataset_path=None, model_config_path: str = None
):
    """
    Check that a model can be loaded, trained and used to make predictions
    """
    if dataset_path is None:
        dataset = datasets["hydromet_5_filtered"].load()
    else:
        dataset = DataSet.from_csv(dataset_path, FullData)
    train, tests = train_test_generator(dataset, 3, n_test_sets=2)
    context, future, truth = next(tests)
    logger.info("Dataset: ")
    logger.info(dataset.to_pandas())

    if model_config_path is not None:
        model_config = ModelConfiguration.model_validate(yaml.safe_load(open(model_config_path)))
    else:
        model_config = None
    try:
        model_template = get_model_template_from_directory_or_github_url(model_url, ignore_env=use_local_environement)
        model = model_template.get_model(model_config)
        estimator = model()
    except Exception as e:
        logger.error(f"Error while creating model: {e}")
        raise e
    try:
        predictor = estimator.train(train)
    except Exception as e:
        logger.error(f"Error while training model: {e}")
        raise e
    try:
        predictions = predictor.predict(context, future)
    except Exception as e:
        logger.error(f"Error while forecasting: {e}")
        raise e
    for location, prediction in predictions.items():
        assert not np.isnan(prediction.samples).any(), (
            f"NaNs in predictions for location {location}, {prediction.samples}"
        )
    context, future, truth = next(tests)
    try:
        predictions = predictor.predict(context, future)
    except Exception as e:
        logger.error(f"Error while forecasting from a future time point: {e}")
        raise e
    for location, prediction in predictions.items():
        assert not np.isnan(prediction.samples).any(), (
            f"NaNs in futuresplit predictions for location {location}, {prediction.samples}"
        )


def serve(seedfile: Optional[str] = None, debug: bool = False, auto_reload: bool = False):
    """
    Start CHAP as a backend server
    """
    from chap_core.rest_api_src.v1.rest_api import main_backend

    logger.info("Running chap serve")

    if seedfile is not None:
        data = json.load(open(seedfile))
    else:
        data = None

    main_backend(data, auto_reload=auto_reload)


def write_open_api_spec(out_path: str):
    """
    Write the OpenAPI spec to a file
    """
    from chap_core.rest_api_src.v1.rest_api import get_openapi_schema

    schema = get_openapi_schema()
    with open(out_path, "w") as f:
        json.dump(schema, f, indent=4)


def test(**base_kwargs):
    """
    Simple test-command to check that the chap command works
    """
    initialize_logging()

    logger.debug("Debug message")
    logger.info("Info message")


def plot_dataset(data_filename: Path, plot_name: str = "standardized_feature_plot"):
    dataset_plot_registry = {
        "standardized_feature_plot": StandardizedFeaturePlot,
        "season_plot": SeasonCorrelationBarPlot,
    }
    plot_cls = dataset_plot_registry.get(plot_name, StandardizedFeaturePlot)
    df = pd.read_csv(data_filename)
    plotter = plot_cls(df)
    fig = plotter.plot()
    fig.show()


def plot_backtest(input_file: Path, output_file: Path, plot_type: str = "backtest_plot_1"):
    """
    Generate a backtest plot from evaluation data and save to file.

    Args:
        input_file: Path to NetCDF file containing evaluation data (from evaluate2)
        output_file: Path to output file (supports .html, .png, .svg, .pdf)
        plot_type: Type of plot to generate. Options: backtest_plot_1, evaluation_plot,
                   ratio_of_samples_above_truth
    """
    from chap_core.assessment.backtest_plots.backtest_plot_1 import BackTestPlot1
    from chap_core.assessment.backtest_plots.sample_bias_plot import RatioOfSamplesAboveTruthBacktestPlot
    from chap_core.assessment.evaluation import Evaluation
    from chap_core.plotting.backtest_plot import EvaluationBackTestPlot

    backtest_plots_registry = {
        "backtest_plot_1": BackTestPlot1,
        "evaluation_plot": EvaluationBackTestPlot,
        "ratio_of_samples_above_truth": RatioOfSamplesAboveTruthBacktestPlot,
    }

    if plot_type not in backtest_plots_registry:
        available = ", ".join(backtest_plots_registry.keys())
        raise ValueError(f"Unknown plot type: {plot_type}. Available: {available}")

    logger.info(f"Loading evaluation from {input_file}")
    evaluation = Evaluation.from_file(input_file)

    logger.info(f"Generating {plot_type} plot")
    plot_class = backtest_plots_registry[plot_type]

    # Use from_evaluation for EvaluationBackTestPlot to preserve historical observations
    if plot_type == "evaluation_plot":
        plotter = plot_class.from_evaluation(evaluation)
    else:
        backtest = evaluation.to_backtest()
        plotter = plot_class.from_backtest(backtest)

    chart = plotter.plot()

    output_path = Path(output_file)
    suffix = output_path.suffix.lower()

    logger.info(f"Saving plot to {output_file}")
    if suffix == ".html":
        chart.save(str(output_path))
    elif suffix in (".png", ".svg", ".pdf"):
        chart.save(str(output_path))
    elif suffix == ".json":
        with open(output_path, "w") as f:
            json.dump(chart.to_dict(format="vega"), f, indent=2)
    else:
        raise ValueError(f"Unsupported output format: {suffix}. Use .html, .png, .svg, .pdf, or .json")

    logger.info(f"Plot saved to {output_file}")


def generate_pdf_report(input_file: Path, output_file: Path):
    """
    Generate old-style matplotlib PDF report from a stored backtest.

    Creates a multi-page PDF with one page per location/forecast period,
    showing historical observations and forecast distributions with quantiles.

    Args:
        input_file: Path to NetCDF file containing evaluation data (from evaluate2)
        output_file: Path to output PDF file
    """
    from chap_core.assessment.evaluation import Evaluation
    from chap_core.assessment.prediction_evaluator import generate_pdf_from_evaluation

    logger.info(f"Loading evaluation from {input_file}")
    evaluation = Evaluation.from_file(input_file)

    logger.info(f"Generating PDF report to {output_file}")
    generate_pdf_from_evaluation(evaluation, str(output_file))

    logger.info(f"PDF report saved to {output_file}")


def export_metrics(
    input_files: list[Path],
    output_file: Path,
    metric_ids: Optional[list[str]] = None,
):
    """
    Export metrics from multiple backtest files to CSV.

    Reads NetCDF evaluation files (from evaluate2) and computes aggregate metrics,
    outputting a CSV file with evaluations as rows and metrics as columns.

    Args:
        input_files: List of paths to NetCDF evaluation files
        output_file: Path to output CSV file
        metric_ids: Optional list of metric IDs to compute. If None, all aggregate metrics are computed.
    """
    from chap_core.assessment.evaluation import Evaluation
    from chap_core.assessment.metrics import available_metrics

    # Get list of aggregate metrics
    aggregate_metric_ids = [
        metric_id for metric_id, metric_cls in available_metrics.items() if metric_cls().is_full_aggregate()
    ]

    # Filter to requested metrics (if specified)
    if metric_ids is not None:
        invalid_ids = set(metric_ids) - set(aggregate_metric_ids)
        if invalid_ids:
            available = ", ".join(aggregate_metric_ids)
            raise ValueError(f"Invalid metric IDs: {invalid_ids}. Available aggregate metrics: {available}")
        metrics_to_compute = metric_ids
    else:
        metrics_to_compute = aggregate_metric_ids

    results = []

    for input_file in input_files:
        logger.info(f"Processing {input_file}")

        # Load file metadata directly from xarray to get model info
        ds = xr.open_dataset(input_file)
        model_name = ds.attrs.get("model_name", "")
        model_version = ds.attrs.get("model_version", "")
        ds.close()

        # Load evaluation and compute metrics
        evaluation = Evaluation.from_file(input_file)
        flat_data = evaluation.to_flat()

        row = {
            "filename": input_file.name,
            "model_name": model_name,
            "model_version": model_version,
        }

        for metric_id in metrics_to_compute:
            metric_cls = available_metrics[metric_id]
            metric = metric_cls()
            metric_df = metric.get_metric(flat_data.observations, flat_data.forecasts)
            if len(metric_df) == 1:
                row[metric_id] = float(metric_df["metric"].iloc[0])
            else:
                logger.warning(f"Metric {metric_id} returned {len(metric_df)} rows, expected 1. Skipping.")
                row[metric_id] = None

        results.append(row)

    # Create DataFrame and write to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    logger.info(f"Metrics exported to {output_file}")


def register_commands(app):
    """Register utility commands with the CLI app."""
    app.command()(sanity_check_model)
    app.command()(serve)
    app.command()(write_open_api_spec)
    app.command()(test)
    app.command()(plot_dataset)
    app.command()(plot_backtest)
    app.command()(generate_pdf_report)
    app.command()(export_metrics)
