import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session
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
    summary="List available metric plot types",
)
def get_avilable_metric_plots(backtest_id: int):
    """Return every metric-plot visualization registered in the metric-plots registry.

    The ``backtest_id`` is accepted for symmetry with the per-plot endpoint but does not
    filter the result; the list is the same for every backtest.
    """
    return [
        VisualizationInfo(id=p["id"], display_name=p["name"], description=p["description"]) for p in list_metric_plots()
    ]


class VisualizationParams(DBModel):
    metric_id: int


class MetricInfo(DBModel):
    id: str
    display_name: str
    description: str = ""


@router.get(
    "/metrics/{backtest_id}",
    response_model=list[MetricInfo],
    summary="List available metrics for a backtest",
)
def get_available_metrics(backtest_id: int):
    """Return every metric registered in ``available_metrics`` that can be applied to the given backtest.

    All metrics support detailed-level visualization. The ``backtest_id`` is accepted for
    symmetry with the per-metric plot endpoint but does not filter the result.
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
    """Compute the named metric across the backtest forecasts and render it with the named plot.

    Returns the plot specification as a JSON response (Vega/Vega-Lite shape, depending on
    the plot class). Returns 404 if the backtest or visualization is unknown, and 400 if
    the metric id is not registered.
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
    id: str
    display_name: str
    description: str = ""


@router.get(
    "/dataset-plots/",
    response_model=list[DatasetPlotType],
    summary="List available dataset plot types",
)
def list_dataset_plot_types():
    """Return every dataset-plot visualization registered in the dataset-plots registry."""
    plots = list_dataset_plots()
    return [
        DatasetPlotType(id=plot["id"], display_name=plot["name"], description=plot["description"]) for plot in plots
    ]


@router.get(
    "/dataset-plots/{visualization_name}/{dataset_id}",
    summary="Render a dataset plot",
)
def generate_data_plots(visualization_name: str, dataset_id: int, session: Session = Depends(get_session)):
    """Render the named dataset visualization for the given dataset and return the plot specification as JSON.

    Returns 404 if the dataset or visualization is unknown; the error message lists the
    registered visualization ids.
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
    id: str
    display_name: str
    description: str = ""


@router.get(
    "/backtest-plots/",
    response_model=list[BacktestPlotType],
    summary="List available backtest plot types",
)
def list_backtest_plot_types():
    """Return every backtest-plot visualization registered in the backtest-plots registry."""
    plots = list_backtest_plots()
    return [
        BacktestPlotType(id=plot["id"], display_name=plot["name"], description=plot["description"]) for plot in plots
    ]


@router.get(
    "/backtest-plots/{visualization_name}/{backtest_id}",
    summary="Render a backtest plot",
)
def generate_backtest_plots(visualization_name: str, backtest_id: int, session: Session = Depends(get_session)):
    """Render the named backtest visualization for the given backtest and return the Vega plot specification as JSON.

    Returns 404 if the backtest or visualization is unknown; the error message lists the
    registered visualization ids.
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
