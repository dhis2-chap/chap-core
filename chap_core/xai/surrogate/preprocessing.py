import logging
from dataclasses import dataclass

import numpy as np

from .model import DEFAULT_MODEL_TYPE, make_loo_model_factory

logger = logging.getLogger(__name__)

MIN_SAMPLES_FOR_PERMUTATION = 100
MIN_FEATURES_FOR_PERMUTATION = 6


@dataclass
class FilterResult:
    """Result of feature filtering: constant removal + optional permutation selection."""

    X_filtered: np.ndarray
    kept_feature_names: list[str]
    kept_indices: list[int]
    removed_features: list[str]
    constant_features: list[str]
    perm_removed_features: list[str]


def filter_features(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    imputation_rates: dict[str, float],
    model_type: str = DEFAULT_MODEL_TYPE,
    random_state: int = 42,
    imputation_threshold: float = 0.9,
) -> FilterResult:
    """Filter constant, heavily-imputed, and (optionally) noise features.

    Permutation-based selection only runs when the dataset is large enough
    (>= MIN_SAMPLES_FOR_PERMUTATION rows and > MIN_FEATURES_FOR_PERMUTATION features)
    to produce reliable importance estimates.
    """
    constant_feats: list[str] = []
    removed_feats: list[str] = []
    keep_mask: list[bool] = []

    for i, name in enumerate(feature_names):
        is_constant = np.ptp(X[:, i]) == 0
        imp_rate = imputation_rates.get(name, 0.0)
        if is_constant:
            constant_feats.append(name)
        if is_constant or imp_rate >= imputation_threshold:
            removed_feats.append(name)
            keep_mask.append(False)
            if imp_rate >= imputation_threshold:
                logger.warning(
                    "Removing feature '%s' before fitting: %.0f%% imputed.",
                    name,
                    imp_rate * 100,
                )
        else:
            keep_mask.append(True)

    if constant_feats:
        logger.warning(
            "Features with zero variance (constant): %s -- removed before fitting.",
            constant_feats,
        )

    if not any(keep_mask):
        logger.warning("All features would be removed -- keeping originals as fallback.")
        keep_mask = [True] * len(feature_names)
        removed_feats = []

    keep_indices = [i for i, keep in enumerate(keep_mask) if keep]
    X_filtered = X[:, keep_indices]
    kept_names = [feature_names[i] for i in keep_indices]

    # Permutation-based feature selection: only for large datasets with many features
    perm_removed_feats: list[str] = []
    n_features_filtered = X_filtered.shape[1]
    if n_features_filtered > MIN_FEATURES_FOR_PERMUTATION and len(X_filtered) >= MIN_SAMPLES_FOR_PERMUTATION:
        try:
            from sklearn.inspection import permutation_importance
            from sklearn.model_selection import train_test_split

            X_tr, X_val, y_tr, y_val = train_test_split(X_filtered, y, test_size=0.25, random_state=random_state)
            quick_model = make_loo_model_factory(model_type, random_state=random_state, n_samples=len(X_tr))()
            quick_model.fit(X_tr, y_tr)
            perm_result = permutation_importance(
                quick_model,
                X_val,
                y_val,
                n_repeats=10,
                random_state=random_state,
                scoring="neg_mean_squared_error",
            )
            importances = perm_result.importances_mean
            min_keep = min(3, n_features_filtered)
            threshold = importances.mean() - importances.std()
            perm_keep_mask = importances >= threshold
            if perm_keep_mask.sum() < min_keep:
                top_indices = np.argsort(importances)[-min_keep:]
                perm_keep_mask = np.zeros(n_features_filtered, dtype=bool)
                perm_keep_mask[top_indices] = True

            if perm_keep_mask.sum() < n_features_filtered:
                perm_removed_indices = np.where(~perm_keep_mask)[0]
                perm_removed_feats = [kept_names[i] for i in perm_removed_indices]
                perm_kept_indices = np.where(perm_keep_mask)[0]
                X_filtered = X_filtered[:, perm_kept_indices]
                kept_names = [kept_names[i] for i in perm_kept_indices]
                keep_indices = [keep_indices[i] for i in perm_kept_indices]
                logger.info(
                    "Permutation feature selection removed %d noise feature(s): %s",
                    len(perm_removed_feats),
                    perm_removed_feats,
                )
        except Exception as e:
            logger.debug("Permutation feature selection failed: %s", e)

    return FilterResult(
        X_filtered=X_filtered,
        kept_feature_names=kept_names,
        kept_indices=keep_indices,
        removed_features=removed_feats,
        constant_features=constant_feats,
        perm_removed_features=perm_removed_feats,
    )
