import logging
from typing import Any

import numpy as np

from ..types import FeatureAttribution, GlobalExplanation, LocalExplanation
from .base import SurrogateExplainerBase

logger = logging.getLogger(__name__)


class SurrogateLIMEExplainer(SurrogateExplainerBase):
    """
    Same configurable surrogate as SurrogateSHAPExplainer, but uses LIME for attributions.

    This allows the XAI registry to dispatch to the correct explanation method
    without hardcoded checks in the router. The surrogate model (and its type) is
    identical; only the attribution extraction differs.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._lime_explainer_cache: Any | None = None
        self._lime_cached_X_id: int | None = None
        self._lime_cached_X_shape: tuple[int, ...] | None = None

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
        return self.explain_local_lime(
            X=X,
            instance_idx=instance_idx,
            prediction_id=prediction_id,
            org_unit=org_unit,
            period=period,
            feature_actual_values=feature_actual_values,
            top_k=top_k,
            output_statistic=output_statistic,
            actual_forecast_value=actual_forecast_value,
        )

    def explain_global(
        self,
        X: np.ndarray,
        top_k: int = 10,
        random_state: int = 42,
    ) -> GlobalExplanation:
        """Aggregate LIME importances across a sample of instances for global view."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() before explain_global().")

        n_rows = len(X)
        max_instances = min(n_rows, 50)
        rng = np.random.RandomState(random_state)
        indices = rng.choice(n_rows, size=max_instances, replace=False) if n_rows > max_instances else np.arange(n_rows)

        try:
            from lime.lime_tabular import LimeTabularExplainer

            importance_matrix = np.zeros((len(indices), len(self.feature_names)))
            lime_explainer = LimeTabularExplainer(
                training_data=X,
                feature_names=self.feature_names,
                mode="regression",
                discretize_continuous=False,
            )
            global_n_samples = min(1000, max(300, 30 * X.shape[1]))
            for row_idx, data_idx in enumerate(indices):
                exp = lime_explainer.explain_instance(
                    X[data_idx],
                    self.predict,
                    num_features=len(self.feature_names),
                    num_samples=global_n_samples,
                )
                for feat_idx, weight in next(iter(exp.local_exp.values()), []):
                    if 0 <= feat_idx < len(self.feature_names):
                        importance_matrix[row_idx, feat_idx] += float(weight)

            mean_abs = np.mean(np.abs(importance_matrix), axis=0)
            mean_signed = np.mean(importance_matrix, axis=0)
        except ModuleNotFoundError:
            logger.debug("lime package not installed; falling back to permutation importance for global explanation.")
            mean_abs, mean_signed = self._global_importances_fallback(X)

        attributions = []
        for i, name in enumerate(self.feature_names):
            attributions.append(
                FeatureAttribution(
                    feature_name=name,
                    importance=float(mean_abs[i]),
                    direction="positive" if mean_signed[i] >= 0 else "negative",
                )
            )
        attributions.sort(key=lambda a: abs(a.importance), reverse=True)

        return GlobalExplanation(
            top_features=attributions[:top_k],
            n_samples=len(indices),
        )

    def explain_local_lime(
        self,
        X: np.ndarray,
        instance_idx: int,
        prediction_id: int,
        org_unit: str,
        period: str,
        feature_actual_values: dict[str, float],
        top_k: int = 10,
        output_statistic: str = "median",
        n_samples: int | None = None,
        actual_forecast_value: float | None = None,
    ) -> LocalExplanation:
        if not self._is_fitted:
            raise RuntimeError("Call fit() before explain_local_lime().")

        baseline = float(np.mean(self.predict(X)))
        actual = (
            actual_forecast_value
            if actual_forecast_value is not None
            else float(self.predict(X[instance_idx : instance_idx + 1])[0])
        )

        try:
            from lime.lime_tabular import LimeTabularExplainer

            X_f = self._filter_X(X)
            kept_names = self._kept_feature_names if self._kept_feature_names is not None else self.feature_names
            effective_n_samples = n_samples if n_samples is not None else min(2000, max(500, 50 * X_f.shape[1]))
            need_lime_rebuild = (
                self._lime_explainer_cache is None
                or self._lime_cached_X_id != id(X)
                or self._lime_cached_X_shape != X.shape
            )
            if need_lime_rebuild:
                self._lime_cached_X_id = id(X)
                self._lime_cached_X_shape = X.shape
                self._lime_explainer_cache = LimeTabularExplainer(
                    training_data=X_f,
                    feature_names=kept_names,
                    mode="regression",
                    discretize_continuous=False,
                )
            exp = self._lime_explainer_cache.explain_instance(  # type: ignore[union-attr]
                X_f[instance_idx],
                self._model.predict,
                num_features=len(kept_names),
                num_samples=effective_n_samples,
            )
            lime_by_feature: dict[str, float] = {}
            for feat_idx, weight in next(iter(exp.local_exp.values()), []):
                if 0 <= feat_idx < len(kept_names):
                    name = kept_names[feat_idx]
                    lime_by_feature[name] = lime_by_feature.get(name, 0.0) + float(weight)

            attributions = [
                FeatureAttribution(
                    feature_name=name,
                    importance=weight,
                    direction="positive" if weight > 0 else "negative",
                    actual_value=feature_actual_values.get(name),
                    baseline_value=baseline,
                )
                for name, weight in lime_by_feature.items()
            ]
        except ModuleNotFoundError:
            logger.debug("lime package not installed; falling back to occlusion for local explanation.")
            occlusion_values = self._local_importances_fallback(X, instance_idx)
            attributions = [
                FeatureAttribution(
                    feature_name=name,
                    importance=float(occlusion_values[i]),
                    direction="positive" if occlusion_values[i] > 0 else "negative",
                    actual_value=feature_actual_values.get(name),
                    baseline_value=baseline,
                )
                for i, name in enumerate(self.feature_names)
            ]

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
