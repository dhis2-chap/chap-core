from datetime import datetime
from typing import Any, cast

from fastapi import HTTPException
from sqlalchemy.orm.attributes import flag_modified
from sqlmodel import Session, select

from chap_core.database.database import SessionWrapper
from chap_core.database.tables import Prediction
from chap_core.database.xai_tables import PredictionExplanation
from chap_core.rest_api.v1.xai_schemas import (
    GlobalExplanationResponse,
    HorizonSummaryResponse,
    LocalExplanationRequest,
    LocalExplanationResponse,
    ShapBeeswarmPoint,
    ShapBeeswarmResponse,
    XaiMethodRead,
)
from chap_core.xai.forecast_matching import find_forecast_row_index
from chap_core.xai.forecast_utils import forecast_actual_value
from chap_core.xai.method_registry import (
    METHOD_TYPE_LABELS,
    NATIVE_SHAP,
    VISUALIZATION_LABELS,
    XAI_METHODS,
)
from chap_core.xai.responses.native_shap import (
    native_shap_global_response,
)
from chap_core.xai.responses.native_shap_horizon import build_native_shap_horizon_summary
from chap_core.xai.responses.quality import quality_response_dict
from chap_core.xai.responses.stored_views import (
    beeswarm_from_stored,
    build_local_explanation_record,
    explanation_to_response,
    horizon_summary_from_stored,
)
from chap_core.xai.responses.surrogate_horizon import horizon_summary_from_surrogate
from chap_core.xai.surrogate.methods import METHOD_TO_MODEL_TYPE
from chap_core.xai.surrogate.pipeline import SurrogateContext, build_surrogate_data, fit_surrogate_explainer

_XAI_METHOD_DEFINITIONS_BY_NAME = {method["name"]: method for method in XAI_METHODS}
_NON_FEATURE_FIELDS = {"time_period", "period", "date", "location"}


def _get_dataset(session: Session, dataset_id: int) -> Any:
    return SessionWrapper(session=session).get_dataset(dataset_id)


def resolve_feature_names(dataset: Any) -> list[str]:
    try:
        feature_names = [f for f in dataset.field_names() if f not in _NON_FEATURE_FIELDS]
    except Exception:
        feature_names = []
    if not feature_names:
        raise HTTPException(status_code=400, detail="Dataset has no usable covariate fields for XAI")
    return feature_names


def validate_xai_method_name(
    xai_method_name: str,
    *,
    allow_archived: bool = False,
) -> dict[str, Any]:
    method = _XAI_METHOD_DEFINITIONS_BY_NAME.get(xai_method_name)
    if method is None:
        raise HTTPException(status_code=404, detail=f"XAI method '{xai_method_name}' not found")
    if method.get("archived", False) and not allow_archived:
        raise HTTPException(status_code=400, detail=f"XAI method '{xai_method_name}' is archived")
    return method


def resolve_canonical_period(
    forecasts: list[Any],
    org_unit: str,
    period: str,
) -> tuple[int | None, str]:
    instance_idx = find_forecast_row_index(forecasts, org_unit, period)
    canonical_period = forecasts[instance_idx].period if instance_idx is not None else period
    return instance_idx, canonical_period


def load_global_entry(meta_data: dict[str, Any] | None, xai_method: str) -> dict[str, Any] | None:
    entry = (meta_data or {}).get("xai", {}).get("global_by_method", {}).get(xai_method)
    if entry is None:
        return None
    return cast("dict[str, Any]", entry)


def global_response_from_entry(xai_method: str, entry: dict[str, Any]) -> GlobalExplanationResponse:
    computed_at_str = entry.get("computedAt")
    return GlobalExplanationResponse(
        method=xai_method,
        top_features=entry.get("topFeatures", []),
        computed_at=datetime.fromisoformat(computed_at_str) if computed_at_str else None,
        n_samples=entry.get("nSamples", 0),
        stability_score=entry.get("stabilityScore"),
        surrogate_quality=entry.get("surrogateQuality"),
    )


def persist_global_entry(
    session: Session,
    prediction: Prediction,
    xai_method: str,
    global_exp: Any,
    quality: dict[str, Any],
    *,
    commit: bool = True,
) -> None:
    session.refresh(prediction)
    meta_data = dict(prediction.meta_data) if prediction.meta_data else {}
    meta_data.setdefault("xai", {}).setdefault("global_by_method", {})[xai_method] = {
        "topFeatures": [f.model_dump() for f in global_exp.top_features],
        "computedAt": global_exp.computed_at.isoformat(),
        "nSamples": global_exp.n_samples,
        "stabilityScore": global_exp.stability_score,
        "surrogateQuality": quality_response_dict(quality),
    }
    prediction.meta_data = meta_data
    flag_modified(prediction, "meta_data")
    session.add(prediction)
    if commit:
        session.commit()


def build_surrogate_context(
    prediction_id: int,
    forecasts: list[Any],
    dataset: Any,
    feature_names: list[str],
    output_statistic: str,
    xai_method: str,
) -> SurrogateContext:
    model_type = METHOD_TO_MODEL_TYPE.get(xai_method, "auto")
    X, y, groups, imputation_rates, covariate_provenance_rows = build_surrogate_data(
        forecasts, dataset, feature_names, output_statistic
    )
    cache_key = (prediction_id, xai_method, output_statistic)
    explainer = fit_surrogate_explainer(
        X,
        y,
        groups,
        model_type,
        feature_names,
        imputation_rates,
        xai_method,
        cache_key=cache_key,
    )
    return SurrogateContext(
        X=X,
        feature_names=feature_names,
        explainer=explainer,
        covariate_provenance_rows=covariate_provenance_rows,
    )


def build_surrogate_beeswarm_response(
    prediction_id: int,
    output_statistic: str,
    forecasts: list[Any],
    surrogate_context: SurrogateContext,
) -> ShapBeeswarmResponse:
    points: list[ShapBeeswarmPoint] = []
    for i, forecast in enumerate(forecasts):
        actual = forecast_actual_value(forecast.values, output_statistic)
        feature_actual_values = {
            name: float(surrogate_context.X[i, j]) for j, name in enumerate(surrogate_context.feature_names)
        }
        local_exp = surrogate_context.explainer.explain_local(
            X=surrogate_context.X,
            instance_idx=i,
            prediction_id=prediction_id,
            org_unit=forecast.org_unit,
            period=forecast.period,
            feature_actual_values=feature_actual_values,
            top_k=len(surrogate_context.feature_names),
            output_statistic=output_statistic,
            actual_forecast_value=actual,
        )
        attr_by_name = {a.feature_name: a.importance for a in local_exp.feature_attributions}
        for j, feature_name in enumerate(surrogate_context.feature_names):
            points.append(
                ShapBeeswarmPoint(
                    feature_name=feature_name,
                    shap_value=float(attr_by_name.get(feature_name, 0.0)),
                    feature_value=float(surrogate_context.X[i, j]),
                    org_unit=forecast.org_unit,
                    period=forecast.period,
                )
            )

    return ShapBeeswarmResponse(
        prediction_id=prediction_id,
        output_statistic=output_statistic,
        feature_names=surrogate_context.feature_names,
        points=points,
        surrogate_quality=quality_response_dict(surrogate_context.explainer.quality_dict()),
    )


def compute_global_explanation_service(
    session: Session,
    prediction: Prediction,
    prediction_id: int,
    xai_method: str,
    top_k: int,
    output_statistic: str,
) -> GlobalExplanationResponse:
    forecasts = prediction.forecasts
    if not forecasts:
        raise HTTPException(status_code=400, detail="No forecasts found for prediction")

    dataset = _get_dataset(session, prediction.dataset_id)
    feature_names = resolve_feature_names(dataset)

    if xai_method == NATIVE_SHAP:
        resp = native_shap_global_response(prediction_id, prediction, xai_method)
        if resp is None:
            raise HTTPException(status_code=404, detail="No native SHAP data for this prediction")
        return resp

    surrogate_context = build_surrogate_context(
        prediction_id, forecasts, dataset, feature_names, output_statistic, xai_method
    )
    global_exp = surrogate_context.explainer.explain_global(surrogate_context.X, top_k=top_k)
    quality = surrogate_context.explainer.quality_dict()
    persist_global_entry(session, prediction, xai_method, global_exp, quality)

    return GlobalExplanationResponse(
        method=xai_method,
        top_features=global_exp.top_features,
        computed_at=global_exp.computed_at,
        n_samples=global_exp.n_samples,
        stability_score=global_exp.stability_score,
        surrogate_quality=quality_response_dict(quality),
    )


def compute_local_explanation_service(
    session: Session,
    prediction: Prediction,
    prediction_id: int,
    instance_idx: int,
    canonical_period: str,
    request: LocalExplanationRequest,
) -> LocalExplanationResponse:
    all_forecasts = prediction.forecasts
    dataset = _get_dataset(session, prediction.dataset_id)
    feature_names = resolve_feature_names(dataset)

    if request.xai_method == NATIVE_SHAP:
        raise HTTPException(status_code=404, detail="No native SHAP data for this org_unit/period combination")

    surrogate_context = build_surrogate_context(
        prediction_id, all_forecasts, dataset, feature_names, request.output_statistic, request.xai_method
    )
    explainer = surrogate_context.explainer
    target_forecast = all_forecasts[instance_idx]
    actual_value = forecast_actual_value(target_forecast.values, request.output_statistic)
    feature_actual_values = {name: float(surrogate_context.X[instance_idx, i]) for i, name in enumerate(feature_names)}
    local_exp = explainer.explain_local(
        X=surrogate_context.X,
        instance_idx=instance_idx,
        prediction_id=prediction_id,
        org_unit=request.org_unit,
        period=request.period,
        feature_actual_values=feature_actual_values,
        top_k=request.top_k,
        output_statistic=request.output_statistic,
        actual_forecast_value=actual_value,
    )
    quality = explainer.quality_dict()
    covariate_provenance = None
    if surrogate_context.covariate_provenance_rows and instance_idx < len(surrogate_context.covariate_provenance_rows):
        covariate_provenance = surrogate_context.covariate_provenance_rows[instance_idx]
    explanation = build_local_explanation_record(
        prediction_id=prediction_id,
        org_unit=request.org_unit,
        canonical_period=canonical_period,
        xai_method=request.xai_method,
        output_statistic=request.output_statistic,
        top_k=request.top_k,
        local_exp=local_exp,
        quality=quality,
        covariate_provenance=covariate_provenance,
    )
    session.add(explanation)
    session.commit()
    session.refresh(explanation)
    return explanation_to_response(explanation)


def compute_beeswarm_service(
    session: Session,
    prediction: Prediction,
    prediction_id: int,
    output_statistic: str,
    xai_method: str,
) -> ShapBeeswarmResponse:
    forecasts = prediction.forecasts
    if not forecasts:
        raise HTTPException(status_code=400, detail="No forecasts found for prediction")

    dataset = _get_dataset(session, prediction.dataset_id)
    feature_names = resolve_feature_names(dataset)

    if xai_method == NATIVE_SHAP:
        raise HTTPException(status_code=404, detail="SHAP beeswarm not available for native SHAP predictions")

    surrogate_context = build_surrogate_context(
        prediction_id, forecasts, dataset, feature_names, output_statistic, xai_method
    )
    return build_surrogate_beeswarm_response(prediction_id, output_statistic, forecasts, surrogate_context)


def compute_horizon_summary_service(
    session: Session,
    prediction: Prediction,
    prediction_id: int,
    org_unit: str,
    output_statistic: str,
    xai_method: str,
) -> HorizonSummaryResponse:
    forecasts = prediction.forecasts

    if xai_method == NATIVE_SHAP:
        built = build_native_shap_horizon_summary(
            prediction_id, org_unit, output_statistic, meta_data=prediction.meta_data, forecasts=forecasts
        )
        if built is None:
            raise HTTPException(
                status_code=404,
                detail="No native SHAP data for this prediction or org_unit/period alignment failed",
            )
        return HorizonSummaryResponse.model_validate(built)

    dataset = _get_dataset(session, prediction.dataset_id)
    feature_names = resolve_feature_names(dataset)
    surrogate_context = build_surrogate_context(
        prediction_id, forecasts, dataset, feature_names, output_statistic, xai_method
    )

    if not any(fc.org_unit == org_unit for fc in forecasts):
        available = list({fc.org_unit for fc in forecasts})
        raise HTTPException(
            status_code=404,
            detail=f"No forecasts for org_unit={org_unit}. Available: {available[:10]}",
        )

    return horizon_summary_from_surrogate(
        prediction_id,
        org_unit,
        xai_method,
        output_statistic,
        forecasts,
        surrogate_context.X,
        surrogate_context.feature_names,
        surrogate_context.explainer,
    )


def normalize_list_period(
    forecasts: list[Any],
    org_unit: str | None,
    period: str,
) -> str:
    if "_" in period and org_unit and forecasts:
        idx = find_forecast_row_index(forecasts, org_unit, period)
        if idx is not None:
            return str(forecasts[idx].period)
    return period


def fetch_local_explanations_service(
    session: Session,
    prediction_id: int,
    prediction: Prediction,
    org_unit: str | None,
    period: str | None,
    xai_method: str | None,
) -> list[Any]:
    query = select(PredictionExplanation).where(PredictionExplanation.prediction_id == prediction_id)
    if org_unit:
        query = query.where(PredictionExplanation.org_unit == org_unit)
    if period:
        canonical_period = normalize_list_period(prediction.forecasts, org_unit, period)
        query = query.where(PredictionExplanation.period == canonical_period)
    if xai_method:
        query = query.where(PredictionExplanation.method == xai_method)

    explanations = session.exec(query).all()

    return [explanation_to_response(exp) for exp in explanations]


def get_or_compute_local_explanation(
    session: Session,
    prediction: Prediction,
    prediction_id: int,
    instance_idx: int | None,
    canonical_period: str,
    request: LocalExplanationRequest,
) -> Any:
    period_clause = PredictionExplanation.period == canonical_period
    existing = session.exec(
        select(PredictionExplanation).where(
            PredictionExplanation.prediction_id == prediction_id,
            PredictionExplanation.org_unit == request.org_unit,
            period_clause,
            PredictionExplanation.method == request.xai_method,
        )
    ).first()

    if existing and request.force:
        session.delete(existing)
        session.commit()
        existing = None

    if existing:
        return explanation_to_response(existing)

    if instance_idx is None:
        available = list({f.org_unit for f in prediction.forecasts})
        raise HTTPException(
            status_code=404,
            detail=f"No forecast found for org_unit={request.org_unit}. Available: {available[:10]}",
        )

    return compute_local_explanation_service(
        session, prediction, prediction_id, instance_idx, canonical_period, request
    )


def read_beeswarm(
    session: Session,
    prediction_id: int,
    output_statistic: str,
    xai_method: str,
) -> ShapBeeswarmResponse | None:
    stored_query = select(PredictionExplanation).where(
        PredictionExplanation.prediction_id == prediction_id,
        PredictionExplanation.method == xai_method,
    )
    if xai_method != NATIVE_SHAP:
        stored_query = stored_query.where(PredictionExplanation.output_statistic == output_statistic)
    stored = session.exec(stored_query).all()
    if not stored:
        return None
    return beeswarm_from_stored(prediction_id, output_statistic, stored)


def get_or_compute_beeswarm(
    session: Session,
    prediction: Prediction,
    prediction_id: int,
    output_statistic: str,
    xai_method: str,
) -> ShapBeeswarmResponse:
    cached = read_beeswarm(session, prediction_id, output_statistic, xai_method)
    if cached is not None:
        return cached
    return compute_beeswarm_service(session, prediction, prediction_id, output_statistic, xai_method)


def read_horizon_summary(
    session: Session,
    prediction_id: int,
    org_unit: str,
    output_statistic: str,
    xai_method: str,
) -> HorizonSummaryResponse | None:
    stored_query = select(PredictionExplanation).where(
        PredictionExplanation.prediction_id == prediction_id,
        PredictionExplanation.org_unit == org_unit,
        PredictionExplanation.method == xai_method,
    )
    if xai_method != NATIVE_SHAP:
        stored_query = stored_query.where(PredictionExplanation.output_statistic == output_statistic)
    stored = session.exec(stored_query).all()
    if not stored:
        return None
    return horizon_summary_from_stored(prediction_id, org_unit, xai_method, output_statistic, stored)


def get_or_compute_horizon_summary(
    session: Session,
    prediction: Prediction,
    prediction_id: int,
    org_unit: str,
    output_statistic: str,
    xai_method: str,
) -> HorizonSummaryResponse:
    cached = read_horizon_summary(session, prediction_id, org_unit, output_statistic, xai_method)
    if cached is not None:
        return cached
    return compute_horizon_summary_service(session, prediction, prediction_id, org_unit, output_statistic, xai_method)


def build_xai_method_read(definition: dict[str, Any]) -> XaiMethodRead:
    method_type = definition["method_type"]
    visualizations = definition["supported_visualizations"]
    return XaiMethodRead(
        **definition,
        method_type_label=METHOD_TYPE_LABELS.get(method_type, method_type),
        is_native=method_type == "native_shap",
        supported_visualization_labels=[VISUALIZATION_LABELS.get(v, v) for v in visualizations],
    )


def get_or_compute_global_explanation(
    session: Session,
    prediction: Prediction,
    prediction_id: int,
    xai_method: str,
    top_k: int,
    output_statistic: str,
    force: bool,
) -> GlobalExplanationResponse:
    if not force:
        entry = load_global_entry(prediction.meta_data, xai_method)
        if entry:
            return global_response_from_entry(xai_method, entry)
    return compute_global_explanation_service(session, prediction, prediction_id, xai_method, top_k, output_statistic)
