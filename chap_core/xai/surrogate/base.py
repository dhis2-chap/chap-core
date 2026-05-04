"""Shared surrogate fitting/prediction base for SHAP and LIME explainers."""

import logging
from typing import TYPE_CHECKING

import numpy as np

from .model import DEFAULT_MODEL_TYPE, build_surrogate_model, resolve_model_params
from .preprocessing import FilterResult, filter_features
from .training import train_surrogate

if TYPE_CHECKING:
    from .quality import SurrogateQuality
    from .training import TrainingResult

logger = logging.getLogger(__name__)

MIN_SAMPLES_GLOBAL = 10


class SurrogateExplainerBase:
    """
    Fit a configurable sklearn surrogate on stored (features, target) pairs
    and expose predict/filter helpers shared by SHAP and LIME explainers.
    """

    def __init__(
        self,
        feature_names: list[str],
        model_config: dict | None = None,
        random_state: int = 42,
        hyperparams: dict | None = None,
        imputation_rates: dict[str, float] | None = None,
    ):
        self.feature_names = list(feature_names)
        self.model_config = model_config or {}
        self.hyperparams = hyperparams or {}
        self.imputation_rates = imputation_rates or {}
        self._random_state = random_state

        model_type = self.model_config.get("model_type", DEFAULT_MODEL_TYPE)
        if model_type == "auto":
            model_type = DEFAULT_MODEL_TYPE
        params = resolve_model_params(model_type, self.model_config, hyperparams)
        self._model = build_surrogate_model(model_type, params, random_state=random_state)
        self._is_fitted = False
        self.quality: SurrogateQuality | None = None
        self._keep_indices: list[int] | None = None
        self._kept_feature_names: list[str] | None = None
        self._target_transform_method: str | None = None
        self._X_train: np.ndarray | None = None
        self._baseline_prediction: float = 0.0

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        min_samples: int = MIN_SAMPLES_GLOBAL,
        groups: np.ndarray | None = None,
        filter_result: FilterResult | None = None,
    ) -> None:
        if len(X) < min_samples:
            raise ValueError(f"Not enough data to train surrogate: got {len(X)} rows, need at least {min_samples}.")

        if filter_result is not None:
            X_filtered = filter_result.X_filtered
            self._kept_feature_names = list(filter_result.kept_feature_names)
            self._keep_indices = list(filter_result.kept_indices)
            constant_feats = filter_result.constant_features
            removed_feats = filter_result.removed_features
            perm_removed_feats = filter_result.perm_removed_features
        else:
            fr = filter_features(
                X,
                y,
                self.feature_names,
                self.imputation_rates,
                model_type=self.model_config.get("model_type", DEFAULT_MODEL_TYPE),
                random_state=self._random_state,
            )
            X_filtered = fr.X_filtered
            self._kept_feature_names = list(fr.kept_feature_names)
            self._keep_indices = list(fr.kept_indices)
            constant_feats = fr.constant_features
            removed_feats = fr.removed_features
            perm_removed_feats = fr.perm_removed_features

        model_type = self.model_config.get("model_type", DEFAULT_MODEL_TYPE)
        params = resolve_model_params(model_type, self.model_config, self.hyperparams)

        result = train_surrogate(
            X_filtered=X_filtered,
            y=y,
            model_type=model_type,
            params=params,
            groups=groups,
            random_state=self._random_state,
            imputation_rates=self.imputation_rates,
            constant_features=constant_feats,
            removed_features=removed_feats,
            perm_removed_features=perm_removed_feats,
            n_samples=len(X),
        )

        self._model = result.model
        self.quality = result.quality
        self._target_transform_method = result.target_transform_method
        self._baseline_prediction = result.baseline_prediction
        self._X_train = result.X_background

        self._is_fitted = True
        self._post_fit(result, model_type)

    def _post_fit(self, result: "TrainingResult", model_type: str) -> None:
        """Hook for subclasses to run extra setup after the surrogate is trained."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict().")
        return np.asarray(self._model.predict(self._filter_X(X)), dtype=float)

    def quality_dict(self) -> dict:
        return self.quality.to_dict() if self.quality is not None else {}

    def _filter_X(self, X: np.ndarray) -> np.ndarray:
        return X[:, self._keep_indices] if self._keep_indices is not None else X

    def _expand_shap_values(self, sv_filtered: np.ndarray) -> np.ndarray:
        if self._keep_indices is None:
            return sv_filtered
        full = np.zeros((sv_filtered.shape[0], len(self.feature_names)))
        full[:, self._keep_indices] = sv_filtered
        return full

    def _global_importances_fallback(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Permutation-based global feature importances when the primary method is unavailable."""
        from sklearn.inspection import permutation_importance

        X_f = self._filter_X(X)
        y_surrogate = self._model.predict(X_f)
        result = permutation_importance(self._model, X_f, y_surrogate, n_repeats=10, random_state=self._random_state)
        mean_abs_filtered = np.maximum(result.importances_mean, 0.0)
        mean_abs_full = np.zeros(len(self.feature_names))
        if self._keep_indices is not None:
            mean_abs_full[self._keep_indices] = mean_abs_filtered

        mean_pred = float(np.mean(y_surrogate))
        mean_feat = np.mean(X, axis=0)
        mean_signed = np.where(
            np.mean((X - mean_feat) * (y_surrogate[:, None] - mean_pred), axis=0) >= 0,
            mean_abs_full,
            -mean_abs_full,
        )
        return mean_abs_full, mean_signed

    def _local_importances_fallback(self, X: np.ndarray, instance_idx: int) -> np.ndarray:
        """Occlusion-based local attributions when the primary method is unavailable."""
        X_f = self._filter_X(X)
        instance = X_f[instance_idx : instance_idx + 1]
        baseline_pred = float(self._model.predict(instance)[0])
        col_means = np.mean(X_f, axis=0)

        n_feats = X_f.shape[1]
        occluded_batch = np.repeat(instance, n_feats, axis=0)
        for i in range(n_feats):
            occluded_batch[i, i] = col_means[i]
        attributions_filtered = baseline_pred - self._model.predict(occluded_batch)

        full = np.zeros(len(self.feature_names))
        if self._keep_indices is not None:
            full[self._keep_indices] = attributions_filtered
        else:
            full = attributions_filtered
        return full
