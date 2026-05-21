import json
import logging
from functools import partial
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session
from starlette.responses import JSONResponse

from chap_core.assessment.backtest_plots import (
    FacetedBacktestPlot,
    create_plot_from_backtest,
    get_backtest_plot,
    get_backtest_plots_registry,
    list_backtest_plots,
)
from chap_core.assessment.evaluation import Evaluation
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

router_get = partial(router.get, response_model_by_alias=True)  # MAGIC!: This makes the endpoints return camelCase


# List visualizations
@router.get("/metric-plots/{backtest_id}", response_model=list[VisualizationInfo])
def get_avilable_metric_plots(backtest_id: int):
    """
    List available visualizations
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


@router.get("/metrics/{backtest_id}", response_model=list[MetricInfo])
def get_available_metrics(backtest_id: int):
    """
    List available metrics for visualization.

    All metrics support detailed level visualization.
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


@router.get("/metric-plots/{visualization_name}/{backtest_id}/{metric_id}")
def generate_visualization(
    visualization_name: str, backtest_id: int, metric_id: str, session: Session = Depends(get_session)
):
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


@router.get("/dataset-plots/", response_model=list[DatasetPlotType])
def list_dataset_plot_types():
    plots = list_dataset_plots()
    return [
        DatasetPlotType(id=plot["id"], display_name=plot["name"], description=plot["description"]) for plot in plots
    ]


@router.get("/dataset-plots/{visualization_name}/{dataset_id}")
def generate_data_plots(visualization_name: str, dataset_id: int, session: Session = Depends(get_session)):
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


@router.get("/backtest-plots/", response_model=list[BacktestPlotType])
def list_backtest_plot_types():
    plots = list_backtest_plots()
    return [
        BacktestPlotType(id=plot["id"], display_name=plot["name"], description=plot["description"]) for plot in plots
    ]


@router.get("/backtest-plots/{visualization_name}/{backtest_id}")
def generate_backtest_plots(visualization_name: str, backtest_id: int, session: Session = Depends(get_session)):
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


def _get_plotter_and_flat_data(plot_id: str, backtest_id: int, session: Session):
    plot_cls = get_backtest_plot(plot_id)
    if plot_cls is None:
        raise HTTPException(status_code=404, detail=f"Plot {plot_id} not found")

    # Fixed typo: FacetedBacktestPlot (with the 'ed')
    if not issubclass(plot_cls, FacetedBacktestPlot):
        raise HTTPException(status_code=400, detail=f"Plot '{plot_id}' does not support faceting properties")

    # Fetch the model instance using id
    backtest = session.get(Backtest, backtest_id)
    if not backtest:
        raise HTTPException(status_code=404, detail="Backtest not found")

    evaluation = Evaluation.from_backtest(backtest)
    flat_data = evaluation.to_flat()

    # Extract the underlying frames from the returned flat_data container object
    observations = flat_data.observations
    forecasts = flat_data.forecasts
    historical_df = flat_data.historical_observations

    return plot_cls(), observations, forecasts, historical_df


@router.get("/backtest-plots/{visualization_name}/{backtest_id}/facet-coords")
def get_facet_coordinates(visualization_name: str, backtest_id: int, session: Session = Depends(get_session)) -> dict[str, Any]:
    """
    Returns unique structural dimension arrays available for layout faceting grids.
    """
    plotter, observations, forecasts, historical_df = _get_plotter_and_flat_data(visualization_name, backtest_id, session)
    return plotter.facet_coords(observations, forecasts, historical_df)


@router.post("/backtest-plots/{visualization_name}/{backtest_id}/subplot")
def generate_isolated_plots(visualization_name: str, backtest_id: int, facet_coords: dict[str, Any], session: Session = Depends(get_session)) -> dict[str, Any]:
    """
    Filters the source datasets by exact coordinate targets and generates a single Vega schema spec.
    """
    plotter, observations, forecasts, historical_df = _get_plotter_and_flat_data(visualization_name, backtest_id, session)

    chart = plotter.get_subplot(observations, forecasts, facet_coords, historical_df)
    return JSONResponse(chart.to_dict(format="vega"))


@router.get("/backtest-plots/{visualization_name}/{backtest_id}/subplots")
def generate_all_subplots(visualization_name: str, backtest_id: int, session: Session = Depends(get_session)) -> list[dict[str, Any]]:
    """
    Generates a full flat checklist mapping coordinate variations against their respective Vega specs.
    """
    plotter, observations, forecasts, historical_df = _get_plotter_and_flat_data(visualization_name, backtest_id, session)
    coords_matrix = plotter.facet_coords(observations, forecasts, historical_df)

    subplot_tuples = plotter.get_subplots(
        observations,
        forecasts,
        coords=coords_matrix,
        historical_observations=historical_df
    )

    return [
        {
            "key": key,
            "spec": subplot.to_dict(format="vega")
        }
        for key, subplot in subplot_tuples
    ]
