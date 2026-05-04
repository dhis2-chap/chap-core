"""Tests for SHAP additivity guarantees under target transformations."""

from __future__ import annotations

import numpy as np
import pytest

from chap_core.xai.surrogate.shap_explainer import SurrogateSHAPExplainer

shap = pytest.importorskip("shap")


def _make_fitted_explainer(X: np.ndarray, y: np.ndarray) -> SurrogateSHAPExplainer:
    feature_names = [f"f{i}" for i in range(X.shape[1])]
    explainer = SurrogateSHAPExplainer(
        feature_names=feature_names,
        model_config={
            "model_type": "decision_tree",
            "max_depth": 4,
            "min_samples_leaf": 2,
        },
    )
    explainer.fit(X, y)
    return explainer


def test_shap_additivity_holds():
    """SHAP guarantee: baseline + sum(sv[i]) == predict(X[i]) for every row."""
    rng = np.random.default_rng(0)
    n, d = 30, 4
    X = rng.normal(size=(n, d)).astype(float)
    y = np.abs(X[:, 0] * 2.0 + X[:, 1] * 0.5 + 1.0) + rng.normal(scale=0.1, size=n)
    explainer = _make_fitted_explainer(X, y)

    sv = explainer.shap_values_matrix(X)
    baseline = explainer.expected_value
    predicted = explainer.predict(X)

    np.testing.assert_allclose(baseline + np.sum(sv, axis=1), predicted, rtol=1e-5, atol=1e-5)


def test_shap_additivity_holds_with_skewed_target():
    """Additivity holds with a right-skewed target that favours a log1p transform."""
    rng = np.random.default_rng(1)
    n, d = 30, 4
    X = rng.normal(size=(n, d)).astype(float)
    y = np.exp(X[:, 0] * 0.8) * 10 + rng.exponential(scale=0.5, size=n)
    explainer = _make_fitted_explainer(X, y)

    sv = explainer.shap_values_matrix(X)
    baseline = explainer.expected_value
    predicted = explainer.predict(X)

    np.testing.assert_allclose(baseline + np.sum(sv, axis=1), predicted, rtol=1e-4, atol=1e-4)


def test_shap_additivity_holds_when_features_carry_no_signal():
    """Additivity holds when the surrogate learns near-zero attributions for every row."""
    rng = np.random.default_rng(7)
    n, d = 30, 4
    X = rng.normal(size=(n, d)).astype(float)
    # Near-constant target: the model predicts close to the mean for all rows,
    # so SHAP values are near zero, exercising the zero-residual branch.
    y = np.ones(n) * 5.0 + rng.normal(scale=1e-3, size=n)
    explainer = _make_fitted_explainer(X, y)

    sv = explainer.shap_values_matrix(X)
    baseline = explainer.expected_value
    predicted = explainer.predict(X)

    np.testing.assert_allclose(baseline + np.sum(sv, axis=1), predicted, rtol=1e-4, atol=1e-4)
