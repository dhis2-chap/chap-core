from types import SimpleNamespace
from typing import Any

import numpy as np

from ..covariate_fallback import target_signature, year_month_from_any
from ..forecast_matching import find_forecast_row_index, norm_period_id
from ..method_registry import NATIVE_SHAP
from .stored_views import compute_avg_importance


def _find_native_value_row(
    values: list[dict[str, Any]],
    org_unit: str,
    period: str,
) -> dict[str, Any] | None:
    subset_indices = [i for i, v in enumerate(values) if str(v.get("location", "")) == org_unit]
    if not subset_indices:
        return None
    subset_forecasts = [
        SimpleNamespace(org_unit=org_unit, period=str(values[i].get("time_period", ""))) for i in subset_indices
    ]
    j = find_forecast_row_index(subset_forecasts, org_unit, period)
    if j is None or not (0 <= j < len(subset_indices)):
        return None
    global_idx = subset_indices[j]
    matched = values[global_idx]
    native_period = str(matched.get("time_period", ""))
    if norm_period_id(native_period) == norm_period_id(period):
        return matched
    sig = target_signature(period)
    if sig is not None and sig[0] == "month":
        _, target_year, target_month = sig
        ym = year_month_from_any(native_period)
        if ym == (target_year, target_month):
            return matched
    return None


def build_native_shap_horizon_summary(
    prediction_id: int,
    org_unit: str,
    output_statistic: str,
    *,
    meta_data: dict[str, Any] | None,
    forecasts: list[Any],
) -> dict[str, Any] | None:
    native_shap = (meta_data or {}).get(NATIVE_SHAP)
    if not native_shap:
        return None

    feature_names: list[str] = list(native_shap.get("feature_names", []))
    values: list[dict[str, Any]] = list(native_shap.get("values", []))
    if not feature_names or not values:
        return None

    unit_entries = [(i, fc) for i, fc in enumerate(forecasts) if fc.org_unit == org_unit]
    if not unit_entries:
        return None
    unit_entries.sort(key=lambda x: x[1].period)

    default_expected = float(native_shap.get("expected_value", 0.0))
    steps: list[dict[str, Any]] = []
    all_importances: dict[str, list[float]] = {f: [] for f in feature_names}

    for step_num, (_idx, fc) in enumerate(unit_entries, start=1):
        entry = _find_native_value_row(values, org_unit, fc.period)
        if entry is None:
            return None

        shap_vals = entry["shap_values"]
        expected_value = float(entry.get("expected_value", default_expected))
        actual_prediction = expected_value + float(np.sum(shap_vals))

        feat_imps: list[dict[str, Any]] = []
        for i, fname in enumerate(feature_names):
            val = float(shap_vals[i])
            all_importances[fname].append(val)
            feat_imps.append(
                {
                    "feature_name": fname,
                    "importance": abs(val),
                    "direction": "positive" if val >= 0 else "negative",
                }
            )
        feat_imps.sort(key=lambda x: x["importance"], reverse=True)

        steps.append(
            {
                "period": fc.period,
                "target_period": fc.period,
                "forecast_step": step_num,
                "feature_importances": feat_imps,
                "actual_prediction": actual_prediction,
            }
        )

    avg_importance = compute_avg_importance(all_importances)

    return {
        "prediction_id": prediction_id,
        "org_unit": org_unit,
        "method": NATIVE_SHAP,
        "output_statistic": output_statistic,
        "steps": steps,
        "average_importance": avg_importance,
        "surrogate_quality": None,
    }
