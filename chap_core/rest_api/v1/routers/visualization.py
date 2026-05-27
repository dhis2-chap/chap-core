import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Field, Session
from starlette.responses import JSONResponse

from chap_core.assessment.backtest_plots import (
    create_plot_from_backtest,
    get_backtest_plots_registry,
    list_backtest_plots,
)
from chap_core.assessment.metric_plots import get_metric_plots_registry, list_metric_plots
from chap_core.assessment.metrics import available_metrics
from chap_core.database.base_tables import DBModel
from chap_core.database.dataset_tables import DataSet
from chap_core.database.tables import Backtest
from chap_core.plotting.dataset_plot import create_plot_from_dataset, get_dataset_plots_registry, list_dataset_plots
from chap_core.plotting.evaluation_plot import (
    VisualizationInfo,
    make_plot_from_backtest_object,
)
from chap_core.rest_api.v1.routers.dependencies import get_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/visualization", tags=["Visualizations"])


@router.get(
    "/metric-plots/{backtest_id}",
    response_model=list[VisualizationInfo],
    summary="Discover which metric plots are available",
)
def get_avilable_metric_plots(backtest_id: int):
    """List the metric-plot styles you can render against a backtest's forecasts (line chart of CRPS over time, choropleth of MAE, ...).

    Use this to populate a plot picker in a UI or to find out which
    ``/metric-plots/{name}/...`` URLs are valid. The result is the same regardless of
    ``backtest_id`` — the path takes it for symmetry with the render endpoint.
    """
    return [
        VisualizationInfo(id=p["id"], display_name=p["name"], description=p["description"]) for p in list_metric_plots()
    ]


class VisualizationParams(DBModel):
    """Inputs for requesting a metric-aware backtest plot."""

    metric_id: int = Field(description="Primary key of the metric to score against.")


class MetricInfo(DBModel):
    """Catalogue entry for one scoring metric (CRPS, MAE, ...)."""

    id: str = Field(description="Canonical metric identifier used in URLs and request bodies.")
    display_name: str = Field(description="Human-friendly metric name shown in pickers.")
    description: str = Field(default="", description="Short paragraph explaining what the metric measures.")


@router.get(
    "/metrics/{backtest_id}",
    response_model=list[MetricInfo],
    summary="Discover which scoring metrics are available",
)
def get_available_metrics(backtest_id: int):
    """List the metrics you can score a backtest with (CRPS, MAE, ...), with a human-friendly name and description for each.

    Use this to populate a metric picker in a UI before requesting a specific plot. The
    result is the same regardless of ``backtest_id`` — the path takes it for symmetry
    with the render endpoint.
    """
    logger.info(f"Getting available metrics for backtest {backtest_id}")
    logger.info(f"Available metrics: {available_metrics.keys()}")
    return [
        MetricInfo(
            id=metric_id,
            display_name=metric_factory().get_name(),
            description=metric_factory().get_description(),
        )
        for metric_id, metric_factory in available_metrics.items()
    ]


@router.get(
    "/metric-plots/{visualization_name}/{backtest_id}/{metric_id}",
    summary="Render a metric plot for a backtest",
)
def generate_visualization(
    visualization_name: str, backtest_id: int, metric_id: str, session: Session = Depends(get_session)
):
    """Score a backtest with the chosen metric (CRPS, MAE, ...) and render the result as the chosen plot style.

    The response is a Vega/Vega-Lite spec you can hand straight to a frontend renderer.
    404 if the backtest or plot style is unknown; 400 if the metric id is not one of the
    registered metrics.
    """
    backtest = session.get(Backtest, backtest_id)
    if not backtest:
        raise HTTPException(status_code=404, detail="Backtest not found")

    if metric_id not in available_metrics:
        raise HTTPException(status_code=400, detail=f"Metric {metric_id} not found")

    registry = get_metric_plots_registry()
    if visualization_name not in registry:
        raise HTTPException(status_code=404, detail=f"Visualization {visualization_name} not found")

    geojson_str = backtest.dataset.geojson
    geojson = json.loads(geojson_str) if geojson_str else None
    plot_class = registry[visualization_name]
    metric = available_metrics[metric_id]()
    plot_spec = make_plot_from_backtest_object(backtest, plot_class, metric, geojson)
    return JSONResponse(plot_spec)


class DatasetPlotType(DBModel):
    """Catalogue entry for one dataset-level plot style (observation time-series, polygon overlay, ...)."""

    id: str = Field(description="Canonical plot identifier used in URLs.")
    display_name: str = Field(description="Human-friendly plot name shown in pickers.")
    description: str = Field(default="", description="Short paragraph explaining what the plot shows.")


@router.get(
    "/dataset-plots/",
    response_model=list[DatasetPlotType],
    summary="Discover which dataset plots are available",
)
def list_dataset_plot_types():
    """List the visualizations you can render against an imported dataset (observation time-series per region, polygon overlays, ...).

    Use this to populate a plot picker before requesting a specific
    ``/dataset-plots/{name}/{datasetId}`` URL.
    """
    plots = list_dataset_plots()
    return [
        DatasetPlotType(id=plot["id"], display_name=plot["name"], description=plot["description"]) for plot in plots
    ]


@router.get(
    "/dataset-plots/{visualization_name}/{dataset_id}",
    summary="Render a plot of a dataset",
)
def generate_data_plots(visualization_name: str, dataset_id: int, session: Session = Depends(get_session)):
    """Render the chosen visualization for a dataset — used to inspect observations before training, spot gaps in the data, or share a quick view of what got imported.

    The response is a JSON plot spec the frontend can render directly. 404 if the
    dataset or plot style is unknown; the error message lists the registered styles.
    """
    registry = get_dataset_plots_registry()
    if visualization_name not in registry:
        available = ", ".join(registry.keys())
        raise HTTPException(
            status_code=404, detail=f"Visualization {visualization_name} not found. Available: {available}"
        )

    dataset = session.get(DataSet, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    chart = create_plot_from_dataset(visualization_name, dataset)
    return JSONResponse(chart)


class BacktestPlotType(DBModel):
    """Catalogue entry for one backtest-level plot style (per-metric, per-org-unit, ...)."""

    id: str = Field(description="Canonical plot identifier used in URLs.")
    display_name: str = Field(description="Human-friendly plot name shown in pickers.")
    description: str = Field(default="", description="Short paragraph explaining what the plot shows.")


@router.get(
    "/backtest-plots/",
    response_model=list[BacktestPlotType],
    summary="Discover which backtest plots are available",
)
def list_backtest_plot_types():
    """List the visualizations you can render against a backtest's forecasts (forecast vs. actuals per region, calibration plots, ...).

    Use this to populate a plot picker before requesting a specific
    ``/backtest-plots/{name}/{backtestId}`` URL.
    """
    plots = list_backtest_plots()
    return [
        BacktestPlotType(id=plot["id"], display_name=plot["name"], description=plot["description"]) for plot in plots
    ]


@router.get(
    "/backtest-plots/{visualization_name}/{backtest_id}",
    summary="Render a forecast plot for a backtest",
)
def generate_backtest_plots(visualization_name: str, backtest_id: int, session: Session = Depends(get_session)):
    """Render the chosen visualization for a backtest's forecasts — used to assess model performance, identify regions where forecasts diverge from actuals, or share an evaluation result.

    The response is a Vega plot spec the frontend can render directly. Returns 404 if
    the backtest or plot style is unknown; the error message lists the registered
    styles.
    """
    registry = get_backtest_plots_registry()
    if visualization_name not in registry:
        available = ", ".join(registry.keys())
        raise HTTPException(
            status_code=404, detail=f"Visualization {visualization_name} not found. Available: {available}"
        )

    backtest = session.get(Backtest, backtest_id)
    if not backtest:
        raise HTTPException(status_code=404, detail="Backtest not found")

    chart = create_plot_from_backtest(visualization_name, backtest)
    return JSONResponse(chart.to_dict(format="vega"))
