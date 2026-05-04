"""Surrogate-based SHAP explainer for CHAP external models.

Trains an sklearn surrogate on stored (feature, forecast_outcome) pairs and applies SHAP
for feature attributions. A surrogate is needed because CHAP models run in Docker
containers and cannot be called at explanation time.
"""

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from ..types import (
    FeatureAttribution,
    GlobalExplanation,
    LocalExplanation,
)
from .base import SurrogateExplainerBase
from .model import build_shap_explainer

if TYPE_CHECKING:
    from .training import TrainingResult

logger = logging.getLogger(__name__)


def _importance_to_normalized_ranks(mean_abs: np.ndarray) -> np.ndarray:
    order = np.argsort(-mean_abs)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(mean_abs))
    return ranks / max(len(mean_abs) - 1, 1)


class SurrogateSHAPExplainer(SurrogateExplainerBase):
    """
    Fit a configurable sklearn surrogate on stored (features, target) pairs,
    then use a SHAP explainer to produce feature attributions.

    The surrogate model type is controlled by model_config:
      {"model_type": "random_forest", "n_estimators": 200}

    Supported model types are defined in surrogate.registry.SUPPORTED_MODELS.

    SHAP guarantee (local, tree models): predicted_value = baseline + sum(shap_values)
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._shap_explainer = None
        self._shap_batch_cache: np.ndarray | None = None
        self._shap_batch_cached_X_id: int | None = None
        self._shap_batch_cached_X_shape: tuple[int, ...] | None = None

    def _post_fit(self, result: "TrainingResult", model_type: str) -> None:
        shap_base = getattr(result.model, "regressor_", result.model)
        self._shap_explainer = self._try_build_shap_explainer(shap_base, model_type, result.X_background)
        self._shap_batch_cache = None
        self._shap_batch_cached_X_id = None
        self._shap_batch_cached_X_shape = None

    @staticmethod
    def _try_build_shap_explainer(model: Any, model_type: str, X_train: np.ndarray | None = None) -> Any:
        """Build a SHAP explainer, returning None if the shap package is not installed."""
        try:
            return build_shap_explainer(model, model_type, X_train=X_train)
        except ModuleNotFoundError:
            logger.debug("shap package not installed; SHAP explanations will be unavailable.")
            return None

    def shap_values_matrix(self, X: np.ndarray) -> np.ndarray:
        """Compute SHAP values for the given feature matrix. Shape: (n_rows, n_features)."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() before shap_values_matrix().")
        if self._shap_explainer is None:
            raise RuntimeError(
                "SHAP explainer is not available. Install the 'shap' package to use SHAP-based explanations."
            )
        sv = self._shap_explainer.shap_values(self._filter_X(X))
        sv_full = self._expand_shap_values(sv)

        if self._target_transform_method in ("log1p", "yeo_johnson"):
            ev = self._shap_explainer.expected_value
            baseline_t = float(ev[0]) if hasattr(ev, "__len__") else float(ev)

            shap_sum_t = np.sum(sv_full, axis=1)

            if self._target_transform_method == "log1p":
                y_base = float(np.expm1(baseline_t))
            else:
                transformer = getattr(self._model, "transformer_", None)
                if transformer is None:
                    transformer = getattr(self._model, "transformer", None)
                if transformer is None:
                    raise RuntimeError(
                        "Yeo-Johnson target transform was selected but no fitted "
                        "transformer is available on the surrogate model; refusing "
                        "to return SHAP values in transformed space."
                    )
                y_base = float(transformer.inverse_transform(np.array([[baseline_t]], dtype=float)).reshape(-1)[0])
            y_pred = np.asarray(self.predict(X), dtype=float)

            n_features = sv_full.shape[1]
            residual = y_pred - y_base
            scale = np.ones_like(shap_sum_t, dtype=float)
            nonzero = shap_sum_t != 0
            scale[nonzero] = residual[nonzero] / shap_sum_t[nonzero]
            sv_full = sv_full * scale[:, None]

            # When shap_sum_t == 0 the row is all zeros; spread the residual
            # uniformly so baseline + sum(sv_row) == y_pred holds for every row.
            zero_rows = ~nonzero
            if np.any(zero_rows) and n_features > 0:
                sv_full[zero_rows] = residual[zero_rows, None] / n_features

        return sv_full

    @property
    def expected_value(self) -> float:
        """Baseline prediction (E[f(X)]). Falls back to the training mean when SHAP is unavailable."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() before accessing expected_value.")
        if self._shap_explainer is None:
            return self._baseline_prediction
        ev = self._shap_explainer.expected_value
        baseline_t = float(ev[0]) if hasattr(ev, "__len__") else float(ev)

        if self._target_transform_method == "log1p":
            return float(np.expm1(baseline_t))

        if self._target_transform_method == "yeo_johnson":
            transformer = getattr(self._model, "transformer_", None)
            if transformer is None:
                transformer = getattr(self._model, "transformer", None)
            if transformer is None:
                return baseline_t
            inv = transformer.inverse_transform(np.array([[baseline_t]], dtype=float))
            return float(inv.reshape(-1)[0])

        return baseline_t

    def explain_global(
        self,
        X: np.ndarray,
        top_k: int = 10,
        n_bootstrap: int = 20,
        random_state: int = 42,
    ) -> GlobalExplanation:
        if not self._is_fitted:
            raise RuntimeError("Call fit() before explain_global().")

        if self._shap_explainer is not None:
            shap_values = self.shap_values_matrix(X)
            mean_abs = np.mean(np.abs(shap_values), axis=0)
            mean_signed = np.mean(shap_values, axis=0)
        else:
            logger.debug("shap unavailable; using permutation importance for global explanation.")
            mean_abs, mean_signed = self._global_importances_fallback(X)

        mean_feature_values = np.nanmean(X, axis=0)

        ranking_std = self._bootstrap_ranking_std(
            X,
            n_bootstrap,
            np.random.RandomState(random_state),
            shap_values if self._shap_explainer is not None else None,
        )
        stability = float(max(0.0, min(1.0, 1.0 - np.mean(ranking_std))))

        attributions = []
        for i, name in enumerate(self.feature_names):
            attributions.append(
                FeatureAttribution(
                    feature_name=name,
                    importance=float(mean_abs[i]),
                    direction="positive" if mean_signed[i] >= 0 else "negative",
                    baseline_value=None,
                    actual_value=float(mean_feature_values[i]),
                )
            )
        attributions.sort(key=lambda a: abs(a.importance), reverse=True)

        return GlobalExplanation(
            top_features=attributions[:top_k],
            n_samples=len(X),
            stability_score=stability,
        )

    def explain_local(
        self,
        X: np.ndarray,
        instance_idx: int,
        prediction_id: int,
        org_unit: str,
        period: str,
        feature_actual_values: dict[str, float],
        top_k: int = 10,
        output_statistic: str = "median",
        actual_forecast_value: float | None = None,
    ) -> LocalExplanation:
        if not self._is_fitted:
            raise RuntimeError("Call fit() before explain_local().")

        instance = X[instance_idx : instance_idx + 1]
        if self._shap_explainer is not None:
            need_shap_recompute = (
                self._shap_batch_cache is None
                or self._shap_batch_cached_X_id != id(X)
                or self._shap_batch_cached_X_shape != X.shape
            )
            if need_shap_recompute:
                self._shap_batch_cached_X_id = id(X)
                self._shap_batch_cached_X_shape = X.shape
                self._shap_batch_cache = self.shap_values_matrix(X)
            shap_values = self._shap_batch_cache[instance_idx]
        else:
            logger.debug("shap unavailable; using occlusion for local explanation.")
            shap_values = self._local_importances_fallback(X, instance_idx)
        baseline = self.expected_value

        actual = actual_forecast_value if actual_forecast_value is not None else float(self.predict(instance)[0])

        attributions = []
        for i, name in enumerate(self.feature_names):
            sv = float(shap_values[i])
            attributions.append(
                FeatureAttribution(
                    feature_name=name,
                    importance=sv,
                    direction="positive" if sv > 0 else "negative",
                    actual_value=feature_actual_values.get(name),
                    baseline_value=baseline,
                )
            )
        attributions.sort(key=lambda a: abs(a.importance), reverse=True)

        return LocalExplanation(
            prediction_id=prediction_id,
            org_unit=org_unit,
            period=period,
            output_statistic=output_statistic,
            feature_attributions=attributions[:top_k],
            baseline_prediction=baseline,
            actual_prediction=actual,
        )

    def _bootstrap_ranking_std(
        self,
        X: np.ndarray,
        n_bootstrap: int,
        rng: np.random.RandomState,
        sv_full: np.ndarray | None = None,
    ) -> np.ndarray:
        n_rows, n_feats = X.shape
        rank_matrix = np.zeros((n_bootstrap, n_feats))
        if self._shap_explainer is not None:
            if sv_full is None:
                sv_full = self.shap_values_matrix(X)
            for b in range(n_bootstrap):
                idx = rng.choice(n_rows, size=n_rows, replace=True)
                rank_matrix[b] = _importance_to_normalized_ranks(np.mean(np.abs(sv_full[idx]), axis=0))
        else:
            for b in range(n_bootstrap):
                idx = rng.choice(n_rows, size=n_rows, replace=True)
                mean_abs_boot, _ = self._global_importances_fallback(X[idx])
                rank_matrix[b] = _importance_to_normalized_ranks(mean_abs_boot)
        return np.asarray(np.std(rank_matrix, axis=0), dtype=float)
