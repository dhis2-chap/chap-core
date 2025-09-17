import json
import logging
from functools import partial

from fastapi import APIRouter, Depends
from sqlmodel import Session
from starlette.responses import JSONResponse

from chap_core.database.base_tables import DBModel
from chap_core.database.tables import BackTest
from chap_core.plotting.evaluation_plot import MetricByHorizon, MetricByHorizonV2, MetricMap, MetricMapV2, VisualizationInfo, make_plot_from_backtest_object
from chap_core.rest_api.v1.routers.dependencies import get_session
from chap_core.assessment.metrics import available_metrics

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/visualization", tags=["Visualization"])

router_get = partial(router.get, response_model_by_alias=True)  # MAGIC!: This makes the endpoints return camelCase

plot_registry = {cls.visualization_info.id: cls for cls in [MetricByHorizon, MetricMap]}
plot_registry_v2 = {cls.visualization_info.id: cls for cls in [MetricByHorizonV2, MetricMapV2]}


# List visualizations
@router.get("/{backtest_id}", response_model=list[VisualizationInfo])
def list_visualizations(backtest_id: int):
    """
    List available visualizations
    """
    return list(cls.visualization_info for cls in plot_registry.values())

class VisualizationParams(DBModel):
    metric_id: int

class Metric(DBModel):
    id: str
    display_name: str
    description: str = ''

# todo: this metrics list is outdated and not used after transitioning to new metric system
metrics = [Metric(id='crps',
                  display_name='CRPS',
                  description='Checking if the sampled distribuion matches the truth'),
           Metric(id='crps_norm',
                  display_name='CRPS norm',
                  description='Checking'),
           Metric(id='is_within_10th_90th',
                  display_name='Within 10th 90th percentile'),
           Metric(id='is_within_25th_75th',
                  display_name='Within 25th 75th percentile')]


@router.get("/metrics/{backtest_id}", response_model=list[Metric])
def get_available_metrics(backtest_id: int):
    logger.info(f"Getting available metrics for backtest {backtest_id}")
    suitable_metrics = {id: metric for id, metric in available_metrics.items() if metric().gives_highest_resolution()}
    logger.info(f"All available: {available_metrics.keys()}")
    logger.info(f"Suitable: {suitable_metrics.keys()}")
    return [Metric(id=id,
                   display_name=metric.spec.metric_name,
                   description=metric.spec.description) for id, metric in suitable_metrics.items()
                   ]


@router.get("/{visualization_name}/{backtest_id}/{metric_id}")
def generate_visualization(
        visualization_name: str,
        backtest_id: int,
        metric_id: str,
        session: Session = Depends(get_session)):

    backtest = session.get(BackTest, backtest_id)
    geojson = json.loads(backtest.dataset.geojson)
    if not backtest:
        return {"error": "Backtest not found"}

    suitable_metrics = {id: metric for id, metric in available_metrics.items() if metric.gives_highest_resolution()}
    if metric_id not in suitable_metrics:
        return {"error": f"Metric {metric_id} not found or not suitable for visualization"}

    if visualization_name not in plot_registry_v2:
        return {"error": f"Visualization {visualization_name} not found"}

    plot_class = plot_registry_v2[visualization_name]
    plot_spec = make_plot_from_backtest_object(backtest, plot_class, suitable_metrics[metric_id](), geojson)
    return JSONResponse(plot_spec)

    all_metrics = list(backtest.metrics)
    print(all_metrics)
    print(metric_id)
    metrics = [metric for metric in all_metrics
               if metric.metric_id == metric_id]
    plot_spec = plt_class(metrics, geojson).plot_spec()
    return JSONResponse(plot_spec)


