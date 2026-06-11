"""Unit tests for surrogate models."""

import numpy as np

from chap_core.explainability.surrogate import (
    BayesianSurrogate,
    RidgeSurrogate,
    SurrogateResult,
)


def _linear_data(rng: np.random.Generator, n: int = 80, p: int = 4):
    X = rng.normal(size=(n, p))
    # Stretch / truncate the canonical 4-feature weights so the helper accepts
    # any feature count the caller wants.
    base_weights = np.array([3.0, -2.0, 0.5, 0.0])
    true_w = np.resize(base_weights, p)
    y = X @ true_w + 0.05 * rng.normal(size=n)
    return X, y, true_w


class TestRidgeSurrogate:
    def test_fit_returns_self(self):
        rng = np.random.default_rng(0)
        X, y, _ = _linear_data(rng)
        s = RidgeSurrogate()
        out = s.fit(X, y)
        assert out is s

    def test_recovers_dominant_coefficient_signs(self):
        rng = np.random.default_rng(1)
        X, y, true_w = _linear_data(rng, n=500, p=4)
        coefs = RidgeSurrogate(alpha=0.1).fit(X, y).model.coef_
        # Dominant components match in sign; the near-zero one we don't assert on.
        assert np.sign(coefs[0]) == np.sign(true_w[0])
        assert np.sign(coefs[1]) == np.sign(true_w[1])

    def test_predict_returns_ndarray(self):
        rng = np.random.default_rng(2)
        X, y, _ = _linear_data(rng)
        s = RidgeSurrogate().fit(X, y)
        preds = s.predict(X[:5])
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (5,)

    def test_explain_yields_surrogate_result(self):
        rng = np.random.default_rng(3)
        X, y, _ = _linear_data(rng)
        names = ["a", "b", "c", "d"]
        result = RidgeSurrogate().fit(X, y).explain(names)
        assert isinstance(result, SurrogateResult)
        assert list(result.feature_names) == names
        assert result.weighting.shape == (4,)


class TestBayesianSurrogate:
    def test_fit_with_uniform_weight_default(self):
        rng = np.random.default_rng(0)
        X, y, _ = _linear_data(rng)
        # sample_weight defaults to None and should be replaced by uniform 1s.
        s = BayesianSurrogate().fit(X, y)
        assert s.coef_ is not None
        assert s.coef_.shape == (X.shape[1],)
        assert s.uncertainty is not None

    def test_explain_before_fit_raises(self):
        s = BayesianSurrogate()
        try:
            s.explain(["a", "b"])
        except RuntimeError as exc:
            assert "not fitted" in str(exc).lower()
        else:
            raise AssertionError("expected RuntimeError")

    def test_predict_returns_ndarray(self):
        rng = np.random.default_rng(1)
        X, y, _ = _linear_data(rng)
        s = BayesianSurrogate().fit(X, y)
        preds = s.predict(X[:10])
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (10,)

    def test_acquisition_scores_shape(self):
        rng = np.random.default_rng(2)
        X, y, _ = _linear_data(rng, n=100, p=3)
        s = BayesianSurrogate().fit(X, y)
        candidates = rng.normal(size=(7, 3))
        locality = np.ones(7)
        scores = s.acquisition_scores(candidates, locality)
        assert scores.shape == (7,)
        assert np.all(np.isfinite(scores))


class TestSurrogateResult:
    def test_as_sorted_orders_by_absolute_weight(self):
        result = SurrogateResult(
            feature_names=["a", "b", "c"],
            weighting=np.array([0.5, -2.0, 1.0]),
        )
        ordered = result.as_sorted()
        # Most influential first by absolute value: b (-2), c (1), a (0.5)
        assert [name for name, _ in ordered] == ["b", "c", "a"]
