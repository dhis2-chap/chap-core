"""Utility commands for CHAP CLI."""

from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

from cyclopts import Parameter

from chap_core.api_types import RunConfig
from chap_core.cli_endpoints._args import (  # noqa: TC001 — used at runtime via cyclopts get_type_hints()
    DataConfigurationArg,
    DataSourceMappingArg,
    ModelConfigYamlArg,
    ModelNameArg,
    RunConfigArg,
)
from chap_core.cli_endpoints.generated_plot_ids import BACKTEST_PLOT_IDS

if TYPE_CHECKING:
    from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

logger = logging.getLogger(__name__)

PLOT_TYPE_HELP = "Type of plot to generate. Available: " + ", ".join(f'"{plot_id}"' for plot_id in BACKTEST_PLOT_IDS)


@dataclasses.dataclass
class AreaPolygons: ...


def sanity_check_model(
    model_name: Annotated[ModelNameArg, Parameter(alias="--model-url")],
    *,
    dataset_csv: Annotated[
        str | None,
        Parameter(
            alias="--dataset-path",
            help="Path or URL to a dataset CSV to check the model on. Default: a small example dataset.",
        ),
    ] = None,
    model_configuration_yaml: Annotated[ModelConfigYamlArg, Parameter(alias="--model-config-path")] = None,
    data_source_mapping: DataSourceMappingArg = None,
    data_configuration: DataConfigurationArg = None,
    run_config: RunConfigArg = RunConfig(),
    use_local_environment: Annotated[
        bool,
        Parameter(alias="--use-local-environement", help="Same as --run-config.ignore-environment."),
    ] = False,
):
    """Check that a model can be loaded, trained and used to make predictions.

    Inside a Chap project (see ``chap init``), options not given on the command line are read
    from the project, so ``chap sanity-check-model`` alone checks the project's model on its dataset.
    """
    import numpy as np

    from chap_core.assessment.dataset_splitting import train_test_generator
    from chap_core.cli_endpoints._common import (
        check_model_configuration,
        discover_geojson,
        get_configuration,
        get_estimator,
        load_data_configuration,
        load_dataset_from_csv,
        resolve_csv_path,
    )
    from chap_core.file_io.example_data_set import datasets
    from chap_core.models.model_template import ModelTemplate
    from chap_core.models.utils import CHAP_RUNS_DIR

    if dataset_csv is None:
        dataset = datasets["hydromet_5_filtered"].load()
    else:
        column_mapping = None
        if data_source_mapping is not None:
            with open(data_source_mapping) as f:
                column_mapping = json.load(f)
        csv_path, url_geojson_path = resolve_csv_path(dataset_csv)
        dataset = load_dataset_from_csv(
            csv_path,
            url_geojson_path or discover_geojson(csv_path),
            column_mapping,
            load_data_configuration(data_configuration),
        )
    prediction_length = 3
    train, tests = train_test_generator(dataset, prediction_length, n_test_sets=2)
    context, future, _truth = next(tests)

    model_config = get_configuration(model_configuration_yaml)
    template = ModelTemplate.from_directory_or_github_url(
        model_name,
        base_working_dir=run_config.runs_dir or CHAP_RUNS_DIR,
        ignore_env=run_config.ignore_environment or use_local_environment,
        run_dir_type=run_config.run_directory_type,
        is_chapkit_model=run_config.is_chapkit_model,
        as_chapkit=run_config.as_chapkit,
    )
    with template:
        try:
            check_model_configuration(
                model_config, template.model_template_config, model_configuration_yaml, strict=False
            )
            estimator = get_estimator(template, model_config, prediction_length=prediction_length)
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
        context, future, _truth = next(tests)
        try:
            predictions = predictor.predict(context, future)
        except Exception as e:
            logger.error(f"Error while forecasting from a future time point: {e}")
            raise e
        for location, prediction in predictions.items():
            assert not np.isnan(prediction.samples).any(), (
                f"NaNs in futuresplit predictions for location {location}, {prediction.samples}"
            )

    print(f"{model_name} passed the sanity check: it was trained and made predictions twice.")


def write_open_api_spec(out_path: str):
    """
    Write the OpenAPI spec to a file
    """
    from chap_core.rest_api.app import get_openapi_schema

    schema = get_openapi_schema()
    with open(out_path, "w") as f:
        json.dump(schema, f, indent=4)


def _health_checks() -> list[tuple[str, bool, str]]:
    """Probe that chap-core's core runtime imports resolve.

    Returns one ``(label, ok, detail)`` row per check group; ``detail`` names
    the offending imports when ``ok`` is False. Kept side-effect free (no
    network, Docker, or DB access) so it is safe to run anywhere.
    """
    import importlib

    import chap_core

    checks: list[tuple[str, bool, str]] = []

    def _probe(label, loader, items):
        failed = []
        for item in items:
            try:
                loader(item)
            except Exception as exc:
                failed.append(f"{item} ({type(exc).__name__})")
        ok = not failed
        checks.append((label, ok, "OK" if ok else "missing: " + ", ".join(failed)))

    _probe("numpy / pandas / xarray", importlib.import_module, ["numpy", "pandas", "xarray"])
    _probe(
        "public API (data, fetch, ModelTemplateInterface)",
        lambda name: getattr(chap_core, name),
        ["data", "fetch", "ModelTemplateInterface"],
    )

    return checks


def test():
    """Self-diagnostic: report version/environment and verify core imports resolve.

    Exits non-zero if any check fails, so it is usable as an install smoke test.
    """
    import platform
    import sys

    import chap_core

    print(f"chap-core      {chap_core.__version__}")
    print(f"Python         {platform.python_version()}  ({sys.executable})")
    print(f"platform       {sys.platform}")
    print()
    print("imports:")
    checks = _health_checks()
    for label, _ok, detail in checks:
        print(f"  {label} ... {detail}")
    print()

    if all(ok for _, ok, _ in checks):
        print("All checks passed.")
    else:
        print("Some checks FAILED.")
        raise SystemExit(1)


def plot_dataset(
    data_filename: Annotated[Path, Parameter(help="Dataset CSV to plot.")],
    plot_name: Annotated[str, Parameter(help="Type of plot to make.")] = "standardized-feature-plot",
    output_file: Annotated[
        Path | None,
        Parameter(help="Save the plot to this file (.html, .png, .svg or .pdf) instead of showing it."),
    ] = None,
    browser: Annotated[
        bool,
        Parameter(
            help="Open the plot in a web browser. With --no-browser, the plot is served on a local URL "
            "that is printed, until Ctrl-C."
        ),
    ] = True,
):
    """Plot the features of a dataset over time, one panel per location.

    Opens the plot in a web browser, prints a local URL for it with --no-browser, or saves it with
    --output-file.
    """
    from chap_core.plotting.dataset_plot import get_dataset_plots_registry
    from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

    registry = get_dataset_plots_registry()
    if plot_name not in registry:
        available = ", ".join(registry.keys())
        raise ValueError(f"Unknown plot type: {plot_name}. Available: {available}")
    plot_cls = registry[plot_name]
    ds: DataSet = DataSet.from_csv(data_filename)
    plotter = plot_cls.from_dataset(ds)
    fig = plotter.plot()
    if output_file is not None:
        fig.save(str(output_file))
        print(f"Plot saved to {output_file}")
    elif browser:
        fig.show()
    else:
        serve_html(fig.to_html())


def serve_html(html: str) -> None:
    """Serve an HTML page on a free local port and print its URL, until Ctrl-C."""
    from http.server import BaseHTTPRequestHandler, HTTPServer

    content = html.encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, format, *args):  # noqa: A002 — the name BaseHTTPRequestHandler uses
            pass

    with HTTPServer(("127.0.0.1", 0), Handler) as server:
        print(f"Plot at http://127.0.0.1:{server.server_port}/ (Ctrl-C to stop)", flush=True)
        with contextlib.suppress(KeyboardInterrupt):
            server.serve_forever()


def plot_backtest(
    input_file: Annotated[
        Path,
        Parameter(help="Path to NetCDF file containing evaluation data (from eval)"),
    ],
    output_file: Annotated[
        Path,
        Parameter(help="Path to output file (supports .html, .png, .svg, .pdf)"),
    ],
    plot_type: Annotated[
        str,
        Parameter(help=PLOT_TYPE_HELP),
    ] = "evaluation_plot",
):
    """
    Generate a backtest plot from evaluation data and save to file.
    """
    from chap_core.assessment.backtest_plots import (
        create_plot_from_evaluation,
        get_backtest_plots_registry,
    )
    from chap_core.assessment.evaluation import Evaluation

    registry = get_backtest_plots_registry()
    if plot_type not in registry:
        available = ", ".join(registry.keys())
        raise ValueError(f"Unknown plot type: {plot_type}. Available: {available}")

    logger.info(f"Loading evaluation from {input_file}")
    evaluation = Evaluation.from_file(input_file)

    logger.info(f"Generating {plot_type} plot")
    chart = create_plot_from_evaluation(plot_type, evaluation)

    output_path = Path(output_file)
    suffix = output_path.suffix.lower()

    logger.info(f"Saving plot to {output_file}")
    if suffix == ".html" or suffix in (".png", ".svg", ".pdf"):
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
        input_file: Path to NetCDF file containing evaluation data (from eval)
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
    input_files: Annotated[
        list[Path],
        Parameter(consume_multiple=True, help="NetCDF evaluation files, such as `--input-files output/*.nc`"),
    ],
    *,
    output_file: Path,
    metric_ids: list[str] | None = None,
):
    """
    Export metrics from multiple backtest files to CSV.

    Reads NetCDF evaluation files (from eval) and computes aggregate metrics,
    outputting a CSV file with evaluations as rows and metrics as columns.

    Args:
        input_files: List of paths to NetCDF evaluation files
        output_file: Path to output CSV file
        metric_ids: Optional list of metric IDs to compute. If None, all metrics are computed at AGGREGATE level.
    """
    import pandas as pd
    import xarray as xr

    from chap_core.assessment.evaluation import Evaluation
    from chap_core.assessment.metrics import available_metrics, calculate_metrics

    # All unified metrics support global aggregation
    all_metric_ids = list(available_metrics.keys())

    # Filter to requested metrics (if specified)
    if metric_ids is not None:
        invalid_ids = set(metric_ids) - set(all_metric_ids)
        if invalid_ids:
            available = ", ".join(all_metric_ids)
            raise ValueError(f"Invalid metric IDs: {invalid_ids}. Available metrics: {available}")
        metrics_to_compute = metric_ids
    else:
        metrics_to_compute = all_metric_ids

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

        row = {
            "filename": input_file.name,
            "model_name": model_name,
            "model_version": model_version,
        }

        metrics_dict = calculate_metrics(evaluation=evaluation, metric_ids=metrics_to_compute, row=row)
        results.append(metrics_dict)

    # Create DataFrame and write to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    logger.info(f"Metrics exported to {output_file}")


def register_commands(app):
    """Register utility commands with the CLI app."""
    app.command()(sanity_check_model)
    app.command()(write_open_api_spec)
    app.command()(test)
    app.command()(plot_dataset)
    app.command()(plot_backtest)
    app.command()(generate_pdf_report)
    app.command()(export_metrics)
