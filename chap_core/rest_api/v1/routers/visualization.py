import logging
from functools import partial

from fastapi import APIRouter, Depends
from gluonts import pydantic
from sqlmodel import Session
from starlette.responses import JSONResponse

from chap_core.database.tables import BackTest
from chap_core.plotting.evaluation_plot import MetricByHorizon
from chap_core.rest_api.v1.routers.dependencies import get_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/visualization", tags=["Visualization"])

router_get = partial(router.get, response_model_by_alias=True)  # MAGIC!: This makes the endpoints return camelCase


# List visualizations
@router.get("/", response_model=list[str])
def list_visualizations():
    """
    List available visualizations
    """
    return ['metric_by_horizon']

class VisualizationParams(pydantic.BaseModel):

    metric_id: int

@router.get("/{visualization_name}/{backtest_id}/{metric_id}")
def generate_visualization(
        visualization_name: str,
        backtest_id: int,
        metric_id: str,
        session: Session = Depends(get_session)):
    assert visualization_name == 'metric_by_horizon', "Only metric_by_horizon is supported"
    backtest = session.get(BackTest, backtest_id)
    if not backtest:
        return {"error": "Backtest not found"}
    all_metrics = list(backtest.metrics)
    print(all_metrics)
    print(metric_id)
    metrics = [metric for metric in all_metrics
               if metric.metric_id == metric_id]
    print(metrics)
    plot_spec = MetricByHorizon(metrics).plot_spec()
    return JSONResponse(plot_spec)
