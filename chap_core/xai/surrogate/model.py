"""Surrogate model construction, selection, and tuning utilities for XAI."""

import importlib
import inspect
import logging
from collections.abc import Callable
from typing import Any

import numpy as np

from .registry import (
    DEFAULT_MODEL_TYPE,
    SUPPORTED_MODELS,
    get_display_name,
    get_model_info,
    is_model_available,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_MODEL_TYPE",
    "SUPPORTED_MODELS",
    "auto_select_best_model_type",
    "build_shap_explainer",
    "build_surrogate_model",
    "get_display_name",
    "get_model_info",
    "loo_r2",
    "make_loo_model_factory",
    "make_model_factory",
    "resolve_model_params",
    "select_target_transform",
    "wrap_with_transform",
]


def build_surrogate_model(
    model_type: str, params: dict[str, Any], random_state: int = 42, n_samples: int | None = None
):
    """Instantiate an unfitted sklearn-compatible surrogate model.

    When *n_samples* is provided and < 20, early-stopping parameters are
    removed because the validation split would be too small.
    When *n_samples* < 30, tree complexity is clamped to prevent overfitting.
    HistGradientBoostingRegressor defaults to min_samples_leaf=20; with small n
    that forbids any split (constant predictions, zero TreeSHAP).
    """
    info = get_model_info(model_type)
    effective_params = dict(params)

    if n_samples is not None and model_type == "hist_gradient_boosting":
        msl = effective_params.get("min_samples_leaf", 20)
        cap = max(1, n_samples // 3)
        effective_params["min_samples_leaf"] = min(int(msl), cap)

    if n_samples is not None and n_samples < 32:
        for key in ("n_iter_no_change", "validation_fraction", "early_stopping", "early_stopping_rounds"):
            effective_params.pop(key, None)

    if n_samples is not None and n_samples < 30:
        for key, cap in (("max_depth", 4), ("max_leaf_nodes", 15), ("num_leaves", 15)):
            if key in effective_params and effective_params[key] is not None and effective_params[key] != -1:
                effective_params[key] = min(effective_params[key], cap)
        for key in ("n_estimators", "max_iter"):
            if key in effective_params:
                effective_params[key] = min(effective_params[key], 200)

    module_path, class_name = info["class_dotted"].rsplit(".", 1)
    cls = getattr(importlib.import_module(module_path), class_name)

    # Ridge, Lasso, and some linear models don't accept random_state.
    # CatBoost uses random_seed instead of random_state.
    sig_params = inspect.signature(cls).parameters
    if "random_state" in sig_params:
        return cls(random_state=random_state, **effective_params)
    if "random_seed" in sig_params:
        return cls(random_seed=random_state, **effective_params)
    return cls(**effective_params)


def resolve_model_params(
    model_type: str,
    model_config: dict[str, Any],
    hyperparams: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Merge default, config-level, and tuned params into a final params dict.

    Priority (lowest → highest): defaults < model_config overrides < hyperparams.
    The "model_type" key in model_config is ignored here.
    """
    info = get_model_info(model_type)
    params: dict[str, Any] = {**info["default_params"]}
    params.update({k: v for k, v in model_config.items() if k != "model_type"})
    if hyperparams:
        params.update(hyperparams)
    return params


def make_loo_model_factory(model_type: str, random_state: int = 42, n_samples: int | None = None) -> Callable:
    """Return a zero-argument callable that produces a cheap LOO model instance."""
    info = get_model_info(model_type)
    loo_params = info["loo_params"]
    return lambda: build_surrogate_model(model_type, loo_params, random_state=random_state, n_samples=n_samples)


def make_model_factory(
    model_type: str,
    params: dict[str, Any],
    random_state: int = 42,
    max_estimators: int = 400,
    n_samples: int | None = None,
) -> Callable:
    """Return a zero-argument callable that produces a model with *params*.

    To keep LOO runtime bounded, ``n_estimators`` / ``max_iter`` are capped at
    *max_estimators*.  When *n_samples* is provided it is forwarded to
    ``build_surrogate_model`` for small-dataset guards.
    """
    capped = dict(params)
    for key in ("n_estimators", "max_iter"):
        if key in capped:
            capped[key] = min(capped[key], max_estimators)
    return lambda: build_surrogate_model(model_type, capped, random_state=random_state, n_samples=n_samples)


def build_shap_explainer(model, model_type: str, X_train: np.ndarray | None = None):
    """Create a SHAP explainer appropriate for the given surrogate model type.

    For linear models, *X_train* is required as background data.
    For XGBoost/LightGBM, TreeExplainer is used with model_output='raw'.
    """
    import shap

    info = get_model_info(model_type)
    shap_type = info["shap_type"]
    if shap_type == "tree":
        return shap.TreeExplainer(model)
    if shap_type == "linear":
        if X_train is None:
            raise ValueError("X_train is required for linear SHAP explainer")
        return shap.LinearExplainer(model, X_train)
    raise ValueError(f"Unsupported shap_type '{shap_type}' for model_type '{model_type}'")


def select_target_transform(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str,
    params: dict,
    random_state: int,
    n_fit: int,
) -> str | None:
    """Choose the best target transform (None, 'log1p', or 'yeo_johnson') via CV."""
    try:
        from sklearn.compose import TransformedTargetRegressor
        from sklearn.model_selection import cross_val_score as _cvs
        from sklearn.preprocessing import PowerTransformer

        X, y = np.asarray(X), np.asarray(y)
        cv_folds = min(n_fit, 5) if n_fit >= 10 else max(2, n_fit)

        def _make_model():
            return build_surrogate_model(model_type, params, random_state=random_state, n_samples=n_fit)

        r2_improvement_threshold = 0.05

        r2_raw = float(np.mean(_cvs(_make_model(), X, y, cv=cv_folds, scoring="r2")))
        best_r2, best_method = r2_raw, None

        if float(np.min(y)) >= 0.0:
            r2_log = float(
                np.mean(
                    _cvs(
                        TransformedTargetRegressor(regressor=_make_model(), func=np.log1p, inverse_func=np.expm1),
                        X,
                        y,
                        cv=cv_folds,
                        scoring="r2",
                    )
                )
            )
            if r2_log > best_r2 + r2_improvement_threshold:
                best_r2, best_method = r2_log, "log1p"

        if n_fit >= 10:
            r2_yj = float(
                np.mean(
                    _cvs(
                        TransformedTargetRegressor(
                            regressor=_make_model(),
                            transformer=PowerTransformer(method="yeo-johnson", standardize=False),
                        ),
                        X,
                        y,
                        cv=cv_folds,
                        scoring="r2",
                    )
                )
            )
            if r2_yj > best_r2 + r2_improvement_threshold:
                best_r2, best_method = r2_yj, "yeo_johnson"

        return best_method
    except Exception as e:
        logger.debug("Target-transform check failed, using raw y: %s", e)
        return None


def wrap_with_transform(base_model: Any, transform_method: str | None) -> Any:
    """Wrap *base_model* with a TransformedTargetRegressor if *transform_method* is set."""
    if transform_method == "log1p":
        from sklearn.compose import TransformedTargetRegressor

        return TransformedTargetRegressor(regressor=base_model, func=np.log1p, inverse_func=np.expm1)
    if transform_method == "yeo_johnson":
        from sklearn.compose import TransformedTargetRegressor
        from sklearn.preprocessing import PowerTransformer

        return TransformedTargetRegressor(
            regressor=base_model,
            transformer=PowerTransformer(method="yeo-johnson", standardize=False),
        )
    return base_model


def loo_r2(
    X: np.ndarray,
    y: np.ndarray,
    model_factory: Callable,
    groups: np.ndarray | None = None,
) -> tuple[float | None, np.ndarray]:
    """
    Leave-one-out (or leave-one-group-out) R² for a model factory.

    When ``groups`` is provided and has at least 2 unique values, uses
    leave-one-group-out CV (hold out all rows for one org_unit at a time).
    Falls back to plain LOO otherwise.

    Returns ``(r2, loo_preds)`` where ``loo_preds`` is the array of
    held-out predictions (all zeros / NaN when n < 4).
    """
    X, y = np.asarray(X), np.asarray(y)
    n = len(X)
    loo_preds = np.zeros(n)
    if n < 4:
        return None, loo_preds

    use_logo = False
    if groups is not None:
        unique_groups = np.unique(groups)
        if len(unique_groups) >= 2:
            use_logo = True

    if use_logo:
        for g in unique_groups:
            test_mask = groups == g
            train_mask = ~test_mask
            if train_mask.sum() == 0:
                continue
            m = model_factory()
            m.fit(X[train_mask], y[train_mask])
            loo_preds[test_mask] = m.predict(X[test_mask])
    else:
        for i in range(n):
            X_train = np.delete(X, i, axis=0)
            y_train = np.delete(y, i)
            m = model_factory()
            m.fit(X_train, y_train)
            loo_preds[i] = m.predict(X[i : i + 1])[0]

    ss_res = np.sum((y - loo_preds) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return r2, loo_preds


def auto_select_best_model_type(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None = None,
    random_state: int = 42,
) -> list[str]:
    """
    Try all supported surrogate model types with their cheap LOO params and
    return them ranked by LOO-R², best first.  Returns the top-3 types so
    callers can tune multiple candidates and pick the winner.

    Optional models (XGBoost, LightGBM) are included when their packages are
    available, and silently skipped otherwise.

    Falls back to ``[DEFAULT_MODEL_TYPE]`` if no model can be evaluated
    (e.g. too few samples).
    """
    ranked: list[tuple[float, str]] = []

    for model_type, model_info in SUPPORTED_MODELS.items():
        if not model_info.get("auto_eligible", True):
            continue
        if not is_model_available(model_type):
            logger.debug("Auto-select: skipping %s (package not available)", model_type)
            continue
        try:
            factory = make_loo_model_factory(model_type, random_state=random_state, n_samples=len(X))
            r2, _ = loo_r2(X, y, factory, groups=groups)
        except Exception as exc:
            logger.debug("Auto-select: %s failed with %s", model_type, exc)
            continue
        logger.info("Auto-select: %s LOO-R²=%.4f", model_type, r2 if r2 is not None else float("nan"))
        if r2 is not None:
            ranked.append((r2, model_type))

    if not ranked:
        logger.info("Auto-select: no model evaluated, falling back to %s", DEFAULT_MODEL_TYPE)
        return [DEFAULT_MODEL_TYPE]

    ranked.sort(key=lambda t: t[0], reverse=True)
    top = [mt for _, mt in ranked[:3]]
    logger.info("Auto-select: top models %s", top)
    return top
