"""Surrogate model training — model fitting, LOO evaluation, and quality assembly.

Training layer only: produces a fitted model and quality metrics.
Must not import shap, lime, or any explanation-related module.
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from .model import (
    build_surrogate_model,
    get_model_info,
    loo_r2,
    select_target_transform,
    wrap_with_transform,
)
from .quality import SurrogateQuality, compute_surrogate_quality

logger = logging.getLogger(__name__)

MIN_SAMPLES_FOR_TARGET_TRANSFORM = 30


@dataclass
class TrainingResult:
    model: Any
    """Fitted model, possibly wrapped with TransformedTargetRegressor."""
    quality: SurrogateQuality
    target_transform_method: str | None
    baseline_prediction: float
    """Mean of model.predict(X_filtered) over the training set.
    Used as the SHAP fallback expected_value when the shap package is unavailable."""
    X_background: np.ndarray | None
    """X_filtered retained for linear surrogate models that need background data for SHAP.
    None for tree models."""


def train_surrogate(
    X_filtered: np.ndarray,
    y: np.ndarray,
    model_type: str,
    params: dict,
    groups: np.ndarray | None,
    random_state: int,
    imputation_rates: dict[str, float],
    constant_features: list[str],
    removed_features: list[str],
    perm_removed_features: list[str],
    n_samples: int,
) -> TrainingResult:
    """Fit a surrogate model and evaluate it with LOO CV.

    *n_samples* is the number of rows in the original (pre-filter) feature matrix,
    stored in SurrogateQuality for reporting.
    """
    n_fit = len(X_filtered)

    target_transform_method: str | None = None
    if n_fit >= MIN_SAMPLES_FOR_TARGET_TRANSFORM:
        target_transform_method = select_target_transform(X_filtered, y, model_type, params, random_state, n_fit)

    final_model: Any = wrap_with_transform(
        build_surrogate_model(model_type, params, random_state=random_state, n_samples=n_fit),
        target_transform_method,
    )
    final_model.fit(X_filtered, y)

    def loo_factory():
        return wrap_with_transform(
            build_surrogate_model(model_type, params, random_state=random_state, n_samples=n_fit),
            target_transform_method,
        )

    r2_cv, loo_preds = loo_r2(X_filtered, y, loo_factory, groups=groups)

    quality = compute_surrogate_quality(
        X_filtered=X_filtered,
        y=y,
        model=final_model,
        model_type=model_type,
        groups=groups,
        loo_preds=loo_preds,
        r2_cv=r2_cv,
        imputation_rates=imputation_rates,
        constant_features=constant_features,
        removed_features=removed_features,
        perm_removed_features=perm_removed_features,
        target_transform_method=target_transform_method,
        n_samples=n_samples,
    )

    baseline_prediction = float(np.mean(final_model.predict(X_filtered)))

    shap_type = get_model_info(model_type).get("shap_type", "tree")
    X_background = X_filtered if shap_type == "linear" else None

    return TrainingResult(
        model=final_model,
        quality=quality,
        target_transform_method=target_transform_method,
        baseline_prediction=baseline_prediction,
        X_background=X_background,
    )
