import json
import logging
from functools import partial
from typing import Type

from fastapi import APIRouter, Depends
from sqlmodel import Session
from starlette.responses import JSONResponse

from chap_core.assessment.backtest_plots import (
    create_plot_from_backtest,
    get_backtest_plots_registry,
    list_backtest_plots,
)
from chap_core.assessment.metrics import available_metrics
from chap_core.database.base_tables import DBModel
from chap_core.database.dataset_tables import DataSet
from chap_core.database.tables import BackTest
from chap_core.plotting.dataset_plot import create_plot_from_dataset, get_dataset_plots_registry, list_dataset_plots
from chap_core.plotting.evaluation_plot import (
    MetricByHorizonV2Mean,
    MetricMapV2,
    MetricPlotV2,
    VisualizationInfo,
    make_plot_from_backtest_object,
)
from chap_core.rest_api.v1.routers.dependencies import get_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/visualization", tags=["Visualizations"])

router_get = partial(router.get, response_model_by_alias=True)  # MAGIC!: This makes the endpoints return camelCase

# Type annotation for registry - concrete subclasses with visualization_info
metric_plots_registry: dict[str, Type[MetricPlotV2]] = {
    MetricByHorizonV2Mean.visualization_info.id: MetricByHorizonV2Mean,
    MetricMapV2.visualization_info.id: MetricMapV2,
}


# List visualizations
@router.get("/metric-plots/{backtest_id}", response_model=list[VisualizationInfo])
def get_avilable_metric_plots(backtest_id: int):
    """
    List available visualizations
    """
    return list(cls.visualization_info for cls in metric_plots_registry.values())


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
    backtest = session.get(BackTest, backtest_id)
    if not backtest:
        return {"error": "Backtest not found"}

    if metric_id not in available_metrics:
        return {"error": f"Metric {metric_id} not found"}

    if visualization_name not in metric_plots_registry:
        return {"error": f"Visualization {visualization_name} not found"}

    geojson_str = backtest.dataset.geojson
    geojson = json.loads(geojson_str) if geojson_str else None
    plot_class = metric_plots_registry[visualization_name]
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
        return {"error": f"Visualization {visualization_name} not found. Available: {available}"}

    dataset = session.get(DataSet, dataset_id)
    if not dataset:
        return {"error": "Dataset not found"}

    chart = create_plot_from_dataset(visualization_name, dataset)
    return JSONResponse(chart)


class BackTestPlotType(DBModel):
    id: str
    display_name: str
    description: str = ""


@router.get("/backtest-plots/", response_model=list[BackTestPlotType])
def list_backtest_plot_types():
    plots = list_backtest_plots()
    return [
        BackTestPlotType(id=plot["id"], display_name=plot["name"], description=plot["description"]) for plot in plots
    ]


@router.get("/backtest-plots/{visualization_name}/{backtest_id}")
def generate_backtest_plots(visualization_name: str, backtest_id: int, session: Session = Depends(get_session)):
    registry = get_backtest_plots_registry()
    if visualization_name not in registry:
        available = ", ".join(registry.keys())
        return {"error": f"Visualization {visualization_name} not found. Available: {available}"}

    backtest = session.get(BackTest, backtest_id)
    if not backtest:
        return {"error": "Backtest not found"}

    chart = create_plot_from_backtest(visualization_name, backtest)
    return JSONResponse(chart.to_dict(format="vega"))
