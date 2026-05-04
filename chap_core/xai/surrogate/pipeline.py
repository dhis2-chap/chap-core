import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from chap_core.log_config import get_status_logger

from ..covariate_fallback import resolve_covariate_row
from ..method_registry import LIME_AUTO
from .cache import get_cached_surrogate, put_cached_surrogate

logger = logging.getLogger(__name__)

MIN_SAMPLES_FOR_TUNING = 15


@dataclass
class SurrogateContext:
    X: np.ndarray
    feature_names: list[str]
    explainer: Any
    covariate_provenance_rows: list[dict[str, Any]] | None = None


def build_surrogate_data(
    forecasts: list,
    dataset: Any,
    feature_names: list[str],
    output_statistic: str = "median",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float], list[dict[str, Any]]]:
    df = dataset.to_pandas()
    has_location = "location" in df.columns
    period_col = next((c for c in ["time_period", "period", "date"] if c in df.columns), None)

    rows: list[dict] = []
    org_units: list[str] = []
    forecast_value_lists: list[list[float]] = []
    covariate_provenance_rows: list[dict[str, Any]] = []

    for fc in forecasts:
        loc_df = df[df["location"] == fc.org_unit] if has_location else df

        pcol = period_col if period_col is not None and period_col in loc_df.columns else None
        row, prov = resolve_covariate_row(
            loc_df,
            pcol or "",
            feature_names,
            fc.period,
            fc.org_unit,
            df,
        )
        rows.append(row)
        covariate_provenance_rows.append(prov)
        org_units.append(fc.org_unit)
        forecast_value_lists.append(list(fc.values))

    columns = []
    imputation_rates: dict[str, float] = {}
    for name in feature_names:
        col = np.array([r.get(name, np.nan) for r in rows], dtype=float)
        n_nan = int(np.sum(np.isnan(col)))
        imputation_rates[name] = n_nan / len(col) if len(col) > 0 else 0.0
        if np.any(np.isnan(col)):
            fill = float(np.nanmedian(col)) if not np.all(np.isnan(col)) else 0.0
            col = np.where(np.isnan(col), fill, col)
        columns.append(col)
    X = np.column_stack(columns) if columns else np.zeros((len(rows), 1))

    if output_statistic == "mean":
        y = np.array([np.mean(v) for v in forecast_value_lists], dtype=float)
    elif output_statistic.startswith("q"):
        try:
            q = float(output_statistic[1:]) / 100.0
        except ValueError:
            q = 0.5
        y = np.array([np.quantile(v, q) for v in forecast_value_lists], dtype=float)
    else:
        y = np.array([np.median(v) for v in forecast_value_lists], dtype=float)

    unique_orgs = list(dict.fromkeys(org_units))
    org_to_idx = {org: i for i, org in enumerate(unique_orgs)}
    groups = np.array([org_to_idx[org] for org in org_units])

    return X, y, groups, imputation_rates, covariate_provenance_rows


def fit_surrogate_explainer(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_type: str,
    feature_names: list[str],
    imputation_rates: dict[str, float],
    xai_method_name: str = "",
    cache_key: tuple | None = None,
) -> Any:
    from .lime_explainer import SurrogateLIMEExplainer
    from .model import auto_select_best_model_type
    from .preprocessing import filter_features
    from .shap_explainer import SurrogateSHAPExplainer
    from .tuning import select_and_tune_best_model_type, tune_surrogate_hyperparameters

    if cache_key is not None:
        cached = get_cached_surrogate(cache_key)
        if cached is not None:
            logger.info("Returning cached surrogate for key %s", cache_key)
            return cached

    status_logger = get_status_logger()
    is_lime = xai_method_name == LIME_AUTO

    fr = filter_features(
        X,
        y,
        feature_names,
        imputation_rates,
        model_type=model_type,
    )
    X_filtered = fr.X_filtered
    kept_feature_names = fr.kept_feature_names

    status_logger.info("Building surrogate on %d forecasts, features: %s", len(X_filtered), kept_feature_names)

    n_rows = len(X_filtered)
    if n_rows >= 200:
        n_trials = 300
    elif n_rows >= 100:
        n_trials = 200
    elif n_rows >= 30:
        n_trials = 120
    else:
        n_trials = max(15, min(40, n_rows))

    effective_model_type = model_type
    hyperparams: dict = {}
    if model_type == "auto" and len(X_filtered) >= 4:
        if n_rows >= MIN_SAMPLES_FOR_TUNING:
            status_logger.info(
                "Selecting and tuning best surrogate model (LOO + Optuna, %d trials total)...",
                n_trials,
            )
            try:
                effective_model_type, hyperparams = select_and_tune_best_model_type(
                    X_filtered, y, groups=groups, n_trials=n_trials
                )
                status_logger.info("Selected surrogate: %s (tuned CV)", effective_model_type)
                logger.info("Tuned hyperparameters: %s", hyperparams)
            except Exception as e:
                logger.warning("Select+tune failed, falling back to LOO selection: %s", e)
                top = auto_select_best_model_type(X_filtered, y, groups=groups)
                effective_model_type = top[0]
                status_logger.info("Auto-selected surrogate: %s (LOO fallback)", effective_model_type)
        else:
            top = auto_select_best_model_type(X_filtered, y, groups=groups)
            effective_model_type = top[0]
            status_logger.info("Auto-selected surrogate: %s (LOO, dataset too small for tuning)", effective_model_type)
    elif n_rows >= MIN_SAMPLES_FOR_TUNING:
        try:
            status_logger.info(
                "Tuning hyperparameters for %s (Optuna, %d trials)...",
                effective_model_type,
                n_trials,
            )
            hyperparams = tune_surrogate_hyperparameters(
                X_filtered, y, model_type=effective_model_type, groups=groups, n_trials=n_trials
            )
            logger.info("Tuned hyperparameters: %s", hyperparams)
        except Exception as e:
            logger.warning("Hyperparameter tuning failed, using defaults: %s", e)

    status_logger.info("Fitting surrogate model on %d samples...", len(X_filtered))
    explainer_cls = SurrogateLIMEExplainer if is_lime else SurrogateSHAPExplainer
    explainer = explainer_cls(
        feature_names=feature_names,
        model_config={"model_type": effective_model_type},
        hyperparams=hyperparams,
        imputation_rates=imputation_rates,
    )
    explainer.fit(X, y, groups=groups, filter_result=fr)

    if cache_key is not None:
        put_cached_surrogate(cache_key, explainer)

    return explainer
