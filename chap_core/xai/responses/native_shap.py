from typing import Any

from chap_core.rest_api.v1.xai_schemas import GlobalExplanationResponse

from ..method_registry import NATIVE_SHAP


def has_native_shap(prediction: Any) -> bool:
    return bool((prediction.meta_data or {}).get("xai", {}).get("global_by_method", {}).get(NATIVE_SHAP))


def native_shap_global_response(
    prediction_id: int, prediction: Any, xai_method: str
) -> GlobalExplanationResponse | None:
    entry = (prediction.meta_data or {}).get("xai", {}).get("global_by_method", {}).get(xai_method)
    if entry is None:
        return None
    return GlobalExplanationResponse(
        method=xai_method,
        top_features=entry.get("topFeatures", []),
        computed_at=entry.get("computedAt"),
        n_samples=entry.get("nSamples", 0),
        stability_score=entry.get("stabilityScore"),
        surrogate_quality=None,
    )
