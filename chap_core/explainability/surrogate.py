"""Surrogate models for the LIME pipeline.

The surrogate is the *interpretable* model that LIME fits to the
(perturbation, black-box prediction) pairs, weighted by locality. Its
linear coefficients **are** the explanation: one weight per interpretable
feature, read as "how much this feature pushed the prediction up or down,
locally around the input being explained."

Two surrogates are provided:

* :class:`RidgeSurrogate` — plain L2-regularised linear regression. Fast,
  stable, the default.
* :class:`BayesianSurrogate` — Bayesian linear regression that also yields a
  posterior covariance over the weights. The extra uncertainty is what
  ``explain_adaptive`` uses to *acquire* informative perturbations (see
  :meth:`BayesianSurrogate.acquisition_scores`).
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sklearn.linear_model import Ridge


@dataclass
class SurrogateResult:
    """A fitted surrogate's coefficients paired with their feature names — i.e. the explanation."""

    feature_names: Sequence[str]
    weighting: np.ndarray

    def as_sorted(self) -> list[tuple[str, float]]:
        """Return ``(feature_name, coefficient)`` pairs sorted by descending absolute weight."""
        pairs = list(zip(self.feature_names, self.weighting.tolist(), strict=False))
        return sorted(pairs, key=lambda t: -abs(t[1]))


class SurrogateModel(Protocol):
    """Contract for a LIME surrogate: fit on weighted perturbations, expose coefficients, predict.

    ``fit`` takes the perturbation design matrix ``X``, the (log-transformed)
    black-box responses ``y``, and per-perturbation locality
    ``sample_weight``; ``explain`` returns the coefficients as a
    :class:`SurrogateResult`; ``predict`` is used to score the surrogate's
    local fidelity (R²).
    """

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "SurrogateModel": ...

    def explain(self, feature_names: list[str]) -> SurrogateResult: ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...


class RidgeSurrogate:
    """L2-regularised linear regression surrogate (the default).

    Thin wrapper over scikit-learn's :class:`~sklearn.linear_model.Ridge`.
    The ``alpha`` penalty shrinks coefficients toward zero, which stabilises
    the explanation when perturbations are collinear or scarce.
    """

    def __init__(self, alpha: float = 5.0, fit_intercept: bool = True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.model: Ridge = Ridge(alpha=self.alpha, fit_intercept=self.fit_intercept)

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "RidgeSurrogate":
        self.model.fit(X, y, sample_weight=sample_weight)
        return self

    def explain(self, feature_names: list[str]) -> SurrogateResult:
        return SurrogateResult(feature_names=feature_names, weighting=self.model.coef_.copy())

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(X))


class BayesianSurrogate:
    """Bayesian linear regression surrogate with a posterior over the weights.

    Closed-form Bayesian linear regression with a zero-mean Gaussian prior on
    the weights (precision ``prior_precision``) and Gaussian observation noise
    (precision ``noise_precision``); both default to 1.0 as in the paper. On
    top of the usual coefficients it produces a posterior covariance, which
    gives per-coefficient uncertainty (``coef_std_``) and powers the adaptive
    acquisition loop. Sample weights enter as per-observation scaling of the
    noise precision, so locality weighting carries through to the posterior.
    """

    def __init__(self, prior_precision=1.0, noise_precision=1.0):
        # These are set to 1.0 in the paper
        self.prior_prec = prior_precision  # Measure of confidence in our initial guess (of our weights being zero)
        self.noise_prec = noise_precision  # Measure of confidence in black box labels

        self.weights = None
        self.uncertainty = None
        self.coef_ = None
        self.coef_std_ = None
        self.intercept_ = None

        self.X_offset = None
        self.y_offset = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "BayesianSurrogate":
        # Center data TODO: Should have fit_intercept arg?
        if sample_weight is None:
            sample_weight = np.ones(X.shape[0])
        self.X_offset = np.average(X, axis=0, weights=sample_weight)
        self.y_offset = np.average(y, weights=sample_weight)
        Xc = X - self.X_offset
        yc = y - self.y_offset

        # From equation 3 in paper
        A = self.prior_prec * np.eye(X.shape[1])  # Prior information (np.eye gives identity matrix)
        A += self.noise_prec * (
            Xc.T @ (sample_weight[:, None] * Xc)
        )  # Add information from weighted data points, with noise caveat

        sigma = np.linalg.pinv(A)  # Covariance matrix

        m = self.noise_prec * (sigma @ (Xc.T @ (sample_weight * yc)))  # Mean weights
        self.coef_ = m

        variances = np.diag(sigma)
        variances = np.clip(variances, 0.0, None)  # In case floating point math results in tiny negative number
        self.coef_std_ = np.sqrt(
            variances
        )  # Standard deviation is the sqrt of the variance, which you get from the diagonal of the covmat

        self.uncertainty = sigma
        self.intercept_ = self.y_offset - (self.X_offset @ self.coef_)

        return self

    def explain(self, feature_names: list[str]) -> SurrogateResult:
        if self.coef_ is None:
            raise RuntimeError("Surrogate not fitted")
        return SurrogateResult(feature_names=feature_names, weighting=self.coef_.copy())

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Surrogate not fitted")
        return np.asarray(X @ self.coef_ + self.intercept_)

    def acquisition_scores(self, X_candidates: np.ndarray, locality_weights: np.ndarray) -> np.ndarray:
        """Score candidate perturbations by expected information gain (used by adaptive LIME).

        Each candidate's score is its predictive variance under the current
        posterior (the quadratic form ``xᵀ Σ x``) times its locality weight —
        so the adaptive loop prefers perturbations that are both close to the
        original input and currently uncertain.
        """
        if self.uncertainty is None:
            raise RuntimeError("Surrogate not fitted")

        Xc = X_candidates - self.X_offset

        quad = np.einsum("ij,jk,ik->i", Xc, self.uncertainty, Xc)  # Fast einstein summation
        return np.asarray(locality_weights * quad)
