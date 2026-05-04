from collections.abc import Sequence
from typing import Any

import numpy as np

from chap_core.database.xai_tables import PredictionExplanation
from chap_core.rest_api.v1.xai_schemas import (
    AverageImportance,
    HorizonFeatureImportance,
    HorizonStepSummary,
    HorizonSummaryResponse,
    LocalExplanationResponse,
    ShapBeeswarmPoint,
    ShapBeeswarmResponse,
)
from chap_core.xai.responses.quality import quality_response_dict


def compute_avg_importance(all_importances: dict[str, list[float]]) -> list[AverageImportance]:
    avg: list[AverageImportance] = []
    for fname, vals in all_importances.items():
        mean_signed = float(np.mean(vals)) if vals else 0.0
        mean_abs = float(np.mean(np.abs(vals))) if vals else 0.0
        avg.append(
            AverageImportance(
                feature_name=fname,
                mean_abs_importance=mean_abs,
                mean_signed_importance=mean_signed,
                direction="positive" if mean_signed >= 0 else "negative",
            )
        )
    avg.sort(key=lambda x: x.mean_abs_importance, reverse=True)
    return avg


def explanation_to_response(exp: PredictionExplanation) -> LocalExplanationResponse:
    result = exp.result or {}
    return LocalExplanationResponse(
        id=exp.id,
        prediction_id=exp.prediction_id,
        org_unit=exp.org_unit,
        period=exp.period,
        method=exp.method,
        output_statistic=exp.output_statistic,
        feature_attributions=result.get("feature_attributions", []),
        baseline_prediction=result.get("baseline_prediction", 0),
        actual_prediction=result.get("actual_prediction", 0),
        computed_at=exp.created or None,
        status=exp.status,
        surrogate_quality=result.get("surrogate_quality"),
        covariate_provenance=result.get("covariate_provenance"),
    )


def build_local_explanation_record(
    prediction_id: int,
    org_unit: str,
    canonical_period: str,
    xai_method: str,
    output_statistic: str,
    top_k: int,
    local_exp: Any,
    quality: dict[str, Any],
    covariate_provenance: dict[str, Any] | None,
) -> PredictionExplanation:
    return PredictionExplanation(
        prediction_id=prediction_id,
        org_unit=org_unit,
        period=canonical_period,
        method=xai_method,
        output_statistic=output_statistic,
        params={"top_k": top_k},
        result={
            "feature_attributions": [f.model_dump() for f in local_exp.feature_attributions],
            "baseline_prediction": local_exp.baseline_prediction,
            "actual_prediction": local_exp.actual_prediction,
            "surrogate_quality": quality_response_dict(quality),
            "covariate_provenance": covariate_provenance,
        },
        status="completed",
    )


def beeswarm_from_stored(
    prediction_id: int,
    output_statistic: str,
    explanations: Sequence[PredictionExplanation],
) -> ShapBeeswarmResponse:
    points: list[ShapBeeswarmPoint] = []
    feature_names_seen: list[str] = []
    quality = None

    for exp in explanations:
        result = exp.result or {}
        if quality is None:
            quality = result.get("surrogate_quality")
        for attr in result.get("feature_attributions", []):
            fname = attr.get("feature_name", "")
            if not fname:
                continue
            points.append(
                ShapBeeswarmPoint(
                    feature_name=fname,
                    shap_value=float(attr.get("importance", 0.0)),
                    feature_value=float(attr.get("actual_value") or 0.0),
                    org_unit=exp.org_unit,
                    period=exp.period,
                )
            )
            if fname not in feature_names_seen:
                feature_names_seen.append(fname)

    return ShapBeeswarmResponse(
        prediction_id=prediction_id,
        output_statistic=output_statistic,
        feature_names=feature_names_seen,
        points=points,
        surrogate_quality=quality,
    )


def horizon_summary_from_stored(
    prediction_id: int,
    org_unit: str,
    method: str,
    output_statistic: str,
    stored: Sequence[PredictionExplanation],
) -> HorizonSummaryResponse:
    stored_sorted = sorted(stored, key=lambda e: e.period)
    steps: list[HorizonStepSummary] = []
    all_importances: dict[str, list[float]] = {}
    quality = None

    for step_num, exp in enumerate(stored_sorted, start=1):
        result = exp.result or {}
        if quality is None:
            quality = result.get("surrogate_quality")
        feat_imps: list[HorizonFeatureImportance] = []
        for attr in result.get("feature_attributions", []):
            fname = attr.get("feature_name", "")
            if not fname:
                continue
            val = float(attr.get("importance", 0.0))
            all_importances.setdefault(fname, []).append(val)
            feat_imps.append(
                HorizonFeatureImportance(
                    feature_name=fname,
                    importance=abs(val),
                    direction="positive" if val >= 0 else "negative",
                )
            )
        feat_imps.sort(key=lambda x: x.importance, reverse=True)
        steps.append(
            HorizonStepSummary(
                period=exp.period,
                target_period=exp.period,
                forecast_step=step_num,
                feature_importances=feat_imps,
                actual_prediction=result.get("actual_prediction"),
            )
        )

    avg_importance = compute_avg_importance(all_importances)

    return HorizonSummaryResponse(
        prediction_id=prediction_id,
        org_unit=org_unit,
        method=method,
        output_statistic=output_statistic,
        steps=steps,
        average_importance=avg_importance,
        surrogate_quality=quality,
    )
