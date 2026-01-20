import json
import logging
from functools import partial

from fastapi import APIRouter, Depends
from sqlmodel import Session
from starlette.responses import JSONResponse

from chap_core.assessment.backtest_plots import (
    get_backtest_plots_registry,
    list_backtest_plots,
    create_plot_from_backtest,
)
from chap_core.database.base_tables import DBModel
from chap_core.database.database import SessionWrapper
from chap_core.database.tables import BackTest
from chap_core.plotting.dataset_plot import StandardizedFeaturePlot
from chap_core.plotting.evaluation_plot import (
    MetricByHorizonV2Mean,
    MetricMapV2,
    VisualizationInfo,
    make_plot_from_backtest_object,
)
from chap_core.plotting.season_plot import SeasonCorrelationBarPlot
from chap_core.rest_api.v1.routers.dependencies import get_session
from chap_core.assessment.metrics import available_metrics  # Import from __init__.py

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/visualization", tags=["Visualization"])

router_get = partial(router.get, response_model_by_alias=True)  # MAGIC!: This makes the endpoints return camelCase

metric_plots_registry = {cls.visualization_info.id: cls for cls in [MetricByHorizonV2Mean, MetricMapV2]}


# List visualizations
@router.get("/metric-plots/{backtest_id}", response_model=list[VisualizationInfo])
def get_avilable_metric_plots(backtest_id: int):
    """
    List available visualizations
    """
    return list(cls.visualization_info for cls in metric_plots_registry.values())


class VisualizationParams(DBModel):
    metric_id: int


class Metric(DBModel):
    id: str
    display_name: str
    description: str = ""


@router.get("/metrics/{backtest_id}", response_model=list[Metric])
def get_available_metrics(backtest_id: int):
    logger.info(f"Getting available metrics for backtest {backtest_id}")
    suitable_metrics = {id: metric for id, metric in available_metrics.items() if metric().gives_highest_resolution()}
    logger.info(f"All available: {available_metrics.keys()}")
    logger.info(f"Suitable: {suitable_metrics.keys()}")
    return [
        Metric(id=id, display_name=metric.spec.metric_name, description=metric.spec.description)
        for id, metric in suitable_metrics.items()
    ]


# TODO: this should be renamed to /metric-visualization/
@router.get("/metric-plots/{visualization_name}/{backtest_id}/{metric_id}")
def generate_visualization(
    visualization_name: str, backtest_id: int, metric_id: str, session: Session = Depends(get_session)
):
    backtest = session.get(BackTest, backtest_id)
    geojson = json.loads(backtest.dataset.geojson)
    if not backtest:
        return {"error": "Backtest not found"}

    suitable_metrics = {id: metric for id, metric in available_metrics.items() if metric().gives_highest_resolution()}
    if metric_id not in suitable_metrics:
        return {"error": f"Metric {metric_id} not found or not suitable for visualization"}

    if visualization_name not in metric_plots_registry:
        return {"error": f"Visualization {visualization_name} not found"}

    plot_class = metric_plots_registry[visualization_name]
    plot_spec = make_plot_from_backtest_object(backtest, plot_class, suitable_metrics[metric_id](), geojson)
    return JSONResponse(plot_spec)


@router.get("/dataset-plots/{visualization_name}/{dataset_id}")
def generate_data_plots(visualization_name: str, dataset_id: int, session: Session = Depends(get_session)):
    plots = {
        "standardized-feature-plot": StandardizedFeaturePlot,
        "seasonal-correlation-plot": SeasonCorrelationBarPlot,
    }
    if visualization_name not in plots:
        return {"error": f"Visualization {visualization_name} not found"}

    sw = SessionWrapper(session=session)
    dataset = sw.get_dataset(dataset_id)
    df = dataset.to_pandas()
    plotter_cls = plots.get(visualization_name)
    plotter = plotter_cls.from_pandas(df)
    chart = plotter.plot_spec()
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
