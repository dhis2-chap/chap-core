import json
import logging
from functools import partial

from fastapi import APIRouter, Depends
from gluonts import pydantic
from sqlmodel import Session
from starlette.responses import JSONResponse

from chap_core.database.tables import BackTest
from chap_core.plotting.evaluation_plot import MetricByHorizon, MetricMap
from chap_core.rest_api.v1.routers.dependencies import get_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/visualization", tags=["Visualization"])

router_get = partial(router.get, response_model_by_alias=True)  # MAGIC!: This makes the endpoints return camelCase

plot_registry = {"metric_by_horizon": MetricByHorizon, "metric_map": MetricMap}


# List visualizations
@router.get("/{backtest_id}", response_model=list[str])
def list_visualizations(backtest_id: int):
    """
    List available visualizations
    """
    return list(plot_registry.keys())

class VisualizationParams(pydantic.BaseModel):
    metric_id: int

class Metric(pydantic.BaseModel):
    id: str
    display_name: str
    description: str = ''

metrics = [Metric(id='crps',
                  display_name='CRPS',
                  description='Checking if the sampled distribuion matches the truth'),
           Metric(id='crps_norm',
                  display_name='CRPS norm',
                  description='Checking'),
           Metric(id='is_within_10th_90th',
                  display_name='Within 10th 90th percentile'),
           Metric(id='is_within_25th_75th',
                  display_name='Within 25th 75th percentile'),
           Metric(id='is_within_25th_75th',
                  display_name='Within 25th 75th percentile')]

@router.get("/metrics/{backtest_id}", response_model=list[Metric])
def get_available_metrics(backtest_id: int):
    return metrics


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
    all_metrics = list(backtest.metrics)
    print(all_metrics)
    print(metric_id)
    metrics = [metric for metric in all_metrics
               if metric.metric_id == metric_id]
    plt_class = plot_registry[visualization_name]
    plot_spec = plt_class(metrics, geojson).plot_spec()
    return JSONResponse(plot_spec)
