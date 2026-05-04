"""
XAI (Explainable AI) API endpoints for CHAP.

Provides endpoints for retrieving and computing explanations for predictions.
"""

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Response
from sqlmodel import Session

from chap_core.database.tables import Prediction
from chap_core.database.xai_tables import PredictionExplanation
from chap_core.rest_api.celery_tasks import (
    JOB_NAME_KW,
    JOB_PREDICTION_ID_KW,
    JOB_TYPE_KW,
    JOB_XAI_METHOD_KW,
    CeleryPool,
)
from chap_core.rest_api.data_models import JobResponse
from chap_core.rest_api.v1.xai_schemas import (
    GlobalExplanationResponse,
    HorizonSummaryResponse,
    LocalExplanationRequest,
    LocalExplanationResponse,
    RunExplanationsRequest,
    ShapBeeswarmResponse,
    XaiMethodRead,
)
from chap_core.xai.batch_explanations import run_explanations_task
from chap_core.xai.method_registry import NATIVE_SHAP, SHAP_AUTO
from chap_core.xai.method_registry import XAI_METHODS as XAI_METHOD_DEFINITIONS
from chap_core.xai.responses.native_shap import has_native_shap
from chap_core.xai.responses.stored_views import explanation_to_response
from chap_core.xai.router_services import (
    build_xai_method_read,
    fetch_local_explanations_service,
    get_or_compute_beeswarm,
    get_or_compute_global_explanation,
    get_or_compute_horizon_summary,
    get_or_compute_local_explanation,
    global_response_from_entry,
    load_global_entry,
    read_beeswarm,
    read_horizon_summary,
    resolve_canonical_period,
    validate_xai_method_name,
)

from .dependencies import get_database_url, get_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/xai")
worker: CeleryPool[Any] = CeleryPool()

_XAI_METHODS = [build_xai_method_read(definition) for definition in XAI_METHOD_DEFINITIONS]


def _require_xai_method(xai_method: str = Query(SHAP_AUTO, alias="xaiMethod")) -> str:
    validate_xai_method_name(xai_method)
    return xai_method


@router.get("/methods", response_model=list[XaiMethodRead], response_model_by_alias=True, tags=["XAI"])
def list_xai_methods(
    include_archived: bool = Query(False, alias="includeArchived"),
    prediction_id: int | None = Query(None, alias="predictionId"),
    session: Session = Depends(get_session),
):
    methods = _XAI_METHODS if include_archived else [m for m in _XAI_METHODS if not m.archived]

    if prediction_id is not None:
        prediction = session.get(Prediction, prediction_id)
        has_native = prediction is not None and has_native_shap(prediction)
        if not has_native:
            methods = [m for m in methods if m.name != NATIVE_SHAP]

    return methods


@router.get("/methods/{name}", response_model=XaiMethodRead, response_model_by_alias=True, tags=["XAI"])
def get_xai_method(name: str):
    method = validate_xai_method_name(name, allow_archived=True)
    return build_xai_method_read(method)


@router.post(
    "/predictions/{predictionId}/explanations/run",
    response_model=JobResponse,
    response_model_by_alias=True,
    tags=["XAI"],
)
def run_explanations(
    prediction_id: Annotated[int, Path(alias="predictionId")],
    request: RunExplanationsRequest,
    database_url: str = Depends(get_database_url),
    session: Session = Depends(get_session),
):
    validate_xai_method_name(request.xai_method)
    prediction = session.get(Prediction, prediction_id)
    if prediction is None:
        raise HTTPException(status_code=404, detail="Prediction not found")
    job_name = f"{prediction.name} {request.xai_method}"
    job = worker.queue_db(
        run_explanations_task,
        prediction_id,
        request.xai_method,
        request.output_statistic,
        request.top_k,
        database_url=database_url,
        **{
            JOB_TYPE_KW: "xai_explanations",
            JOB_NAME_KW: job_name,
            JOB_PREDICTION_ID_KW: prediction_id,
            JOB_XAI_METHOD_KW: request.xai_method,
        },
    )
    return JobResponse(id=job.id)


@router.get(
    "/predictions/{predictionId}/global",
    response_model=GlobalExplanationResponse,
    response_model_by_alias=True,
    responses={204: {"description": "No global explanation cached for this prediction and method"}},
    tags=["XAI"],
)
def get_global_explanation(
    prediction_id: Annotated[int, Path(alias="predictionId")],
    xai_method: str = Query(..., alias="xaiMethod"),
    session: Session = Depends(get_session),
):
    prediction = session.get(Prediction, prediction_id)
    if prediction is None:
        raise HTTPException(status_code=404, detail="Prediction not found")

    validate_xai_method_name(xai_method)
    entry = load_global_entry(prediction.meta_data or {}, xai_method)
    if entry:
        return global_response_from_entry(xai_method, entry)
    return Response(status_code=204)


@router.post(
    "/predictions/{predictionId}/global",
    response_model=GlobalExplanationResponse,
    response_model_by_alias=True,
    tags=["XAI"],
)
def compute_global_explanation(
    prediction_id: Annotated[int, Path(alias="predictionId")],
    top_k: int = Query(10, alias="topK"),
    xai_method: str = Depends(_require_xai_method),
    output_statistic: str = Query("median", alias="outputStatistic"),
    force: bool = Query(False),
    session: Session = Depends(get_session),
):
    prediction = session.get(Prediction, prediction_id)
    if prediction is None:
        raise HTTPException(status_code=404, detail="Prediction not found")

    try:
        return get_or_compute_global_explanation(
            session, prediction, prediction_id, xai_method, top_k, output_statistic, force
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error computing global explanation: %s", e)
        raise HTTPException(status_code=500, detail="Error computing explanation") from e


@router.get(
    "/predictions/{predictionId}/local",
    response_model=list[LocalExplanationResponse],
    response_model_by_alias=True,
    tags=["XAI"],
)
def list_local_explanations(
    prediction_id: Annotated[int, Path(alias="predictionId")],
    org_unit: str | None = Query(None, alias="orgUnit"),
    period: str | None = None,
    xai_method: str | None = Query(None, alias="xaiMethod"),
    session: Session = Depends(get_session),
):
    if xai_method is not None:
        validate_xai_method_name(xai_method)
    prediction = session.get(Prediction, prediction_id)
    if prediction is None:
        raise HTTPException(status_code=404, detail="Prediction not found")

    return fetch_local_explanations_service(session, prediction_id, prediction, org_unit, period, xai_method)


@router.post(
    "/predictions/{predictionId}/local",
    response_model=LocalExplanationResponse,
    response_model_by_alias=True,
    tags=["XAI"],
)
def compute_local_explanation(
    prediction_id: Annotated[int, Path(alias="predictionId")],
    request: LocalExplanationRequest,
    session: Session = Depends(get_session),
):
    validate_xai_method_name(request.xai_method)
    prediction = session.get(Prediction, prediction_id)
    if prediction is None:
        raise HTTPException(status_code=404, detail="Prediction not found")

    all_forecasts = prediction.forecasts
    if not all_forecasts:
        raise HTTPException(status_code=400, detail="No forecasts found for prediction")

    instance_idx, canonical_period = resolve_canonical_period(all_forecasts, request.org_unit, request.period)

    try:
        return get_or_compute_local_explanation(
            session, prediction, prediction_id, instance_idx, canonical_period, request
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error computing local explanation: %s", e)
        raise HTTPException(status_code=500, detail="Error computing local explanation") from e


@router.get(
    "/predictions/{predictionId}/shap-beeswarm",
    response_model=ShapBeeswarmResponse,
    response_model_by_alias=True,
    responses={204: {"description": "No SHAP beeswarm cached for this prediction and method"}},
    tags=["XAI"],
)
def get_shap_beeswarm(
    prediction_id: Annotated[int, Path(alias="predictionId")],
    output_statistic: str = Query("median", alias="outputStatistic"),
    xai_method: str = Depends(_require_xai_method),
    session: Session = Depends(get_session),
):
    prediction = session.get(Prediction, prediction_id)
    if prediction is None:
        raise HTTPException(status_code=404, detail="Prediction not found")

    cached = read_beeswarm(session, prediction_id, output_statistic, xai_method)
    if cached is not None:
        return cached
    return Response(status_code=204)


@router.post(
    "/predictions/{predictionId}/shap-beeswarm",
    response_model=ShapBeeswarmResponse,
    response_model_by_alias=True,
    tags=["XAI"],
)
def compute_shap_beeswarm(
    prediction_id: Annotated[int, Path(alias="predictionId")],
    output_statistic: str = Query("median", alias="outputStatistic"),
    xai_method: str = Depends(_require_xai_method),
    session: Session = Depends(get_session),
):
    prediction = session.get(Prediction, prediction_id)
    if prediction is None:
        raise HTTPException(status_code=404, detail="Prediction not found")

    try:
        return get_or_compute_beeswarm(session, prediction, prediction_id, output_statistic, xai_method)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error computing SHAP beeswarm: %s", e)
        raise HTTPException(status_code=500, detail="Error computing beeswarm") from e


@router.get(
    "/predictions/{predictionId}/local/horizon-summary",
    response_model=HorizonSummaryResponse,
    response_model_by_alias=True,
    responses={204: {"description": "No horizon summary cached for this prediction and method"}},
    tags=["XAI"],
)
def get_horizon_summary(
    prediction_id: Annotated[int, Path(alias="predictionId")],
    org_unit: str = Query(..., alias="orgUnit"),
    output_statistic: str = Query("median", alias="outputStatistic"),
    xai_method: str = Depends(_require_xai_method),
    session: Session = Depends(get_session),
):
    prediction = session.get(Prediction, prediction_id)
    if prediction is None:
        raise HTTPException(status_code=404, detail="Prediction not found")

    cached = read_horizon_summary(session, prediction_id, org_unit, output_statistic, xai_method)
    if cached is not None:
        return cached
    return Response(status_code=204)


@router.post(
    "/predictions/{predictionId}/local/horizon-summary",
    response_model=HorizonSummaryResponse,
    response_model_by_alias=True,
    tags=["XAI"],
)
def compute_horizon_summary(
    prediction_id: Annotated[int, Path(alias="predictionId")],
    org_unit: str = Query(..., alias="orgUnit"),
    output_statistic: str = Query("median", alias="outputStatistic"),
    xai_method: str = Depends(_require_xai_method),
    session: Session = Depends(get_session),
):
    prediction = session.get(Prediction, prediction_id)
    if prediction is None:
        raise HTTPException(status_code=404, detail="Prediction not found")

    if not prediction.forecasts:
        raise HTTPException(status_code=400, detail="No forecasts found for prediction")

    try:
        return get_or_compute_horizon_summary(
            session, prediction, prediction_id, org_unit, output_statistic, xai_method
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error computing horizon summary: %s", e)
        raise HTTPException(status_code=500, detail="Error computing horizon summary") from e


@router.get(
    "/predictions/{predictionId}/local/{explanationId}",
    response_model=LocalExplanationResponse,
    response_model_by_alias=True,
    tags=["XAI"],
)
def get_local_explanation(
    prediction_id: Annotated[int, Path(alias="predictionId")],
    explanation_id: Annotated[int, Path(alias="explanationId")],
    session: Session = Depends(get_session),
):
    explanation = session.get(PredictionExplanation, explanation_id)
    if explanation is None or explanation.prediction_id != prediction_id:
        raise HTTPException(status_code=404, detail="Explanation not found")
    return explanation_to_response(explanation)


@router.delete("/predictions/{predictionId}/local/{explanationId}", tags=["XAI"])
def delete_local_explanation(
    prediction_id: Annotated[int, Path(alias="predictionId")],
    explanation_id: Annotated[int, Path(alias="explanationId")],
    session: Session = Depends(get_session),
):
    explanation = session.get(PredictionExplanation, explanation_id)
    if explanation is None or explanation.prediction_id != prediction_id:
        raise HTTPException(status_code=404, detail="Explanation not found")
    session.delete(explanation)
    session.commit()
    return {"message": "deleted"}
