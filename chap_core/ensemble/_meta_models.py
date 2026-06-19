"""Meta-models and CRPS utilities for stacking ensembles."""

from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import minimize, nnls

from chap_core.assessment.metrics.crps import crps_score_unbiased_matrix

logger = logging.getLogger(__name__)


def crps_ensemble(observations: np.ndarray, forecasts: np.ndarray) -> float:
    return crps_score_unbiased_matrix(observations, forecasts)


def _vincentize_samples(X_samples: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    if not X_samples:
        raise ValueError("X_samples must not be empty")
    target_shape = X_samples[0].shape
    for i, s in enumerate(X_samples):
        if s.shape != target_shape:
            raise ValueError(f"Sample shape mismatch: X_samples[0]={target_shape}, X_samples[{i}]={s.shape}")
    stacked = np.stack(X_samples, axis=0)
    sorted_stack = np.sort(stacked, axis=2)
    w = np.asarray(weights, float).reshape(-1, 1, 1)
    return np.asarray(np.sum(w * sorted_stack, axis=0), float)


class NonNegativeMetaModel:
    def __init__(self) -> None:
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> NonNegativeMetaModel:
        coef_raw, _ = nnls(X, y)
        coef = np.asarray(coef_raw, float)
        self.coef_ = coef
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise ValueError("Meta-model not fitted")
        return np.asarray(np.dot(X, self.coef_), float)


class ProbabilisticMetaModel:
    """
    Meta-model for probabilistic ensembles.

    This meta-model combines base model predictive distributions using vincentization:
    for each location/horizon, we sort the samples from each base model
    to approximate their quantile functions Q_i(p), then form the ensemble
    quantiles as
        Q_ens(p) = sum_i w_i Q_i(p),
    where w_i are non-negative weights on the simplex.

    This makes the ensemble distribution invariant to the internal sample
    ordering of each base model, and is sharper than a pure mixture while
    remaining a standard construction in forecast combination.
    """

    def __init__(self, verbose: bool = False) -> None:
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.verbose = verbose

    def fit(self, X_samples: list[np.ndarray], y: np.ndarray) -> ProbabilisticMetaModel:
        target_shape = X_samples[0].shape
        for i, s in enumerate(X_samples):
            if s.shape != target_shape:
                raise ValueError(f"Sample shape mismatch: X_samples[0]={target_shape}, X_samples[{i}]={s.shape}")

        y = np.asarray(y, float).reshape(-1)

        def obj(w: np.ndarray) -> float:
            ens = _vincentize_samples(X_samples, w)
            return crps_score_unbiased_matrix(y, ens)

        n = len(X_samples)
        w0 = np.ones(n) / n

        cons = [
            {"type": "ineq", "fun": lambda w: w},
            {"type": "eq", "fun": lambda w: w.sum() - 1.0},
        ]

        res = minimize(
            obj,
            w0,
            method="SLSQP",
            constraints=cons,
            options={"ftol": 1e-9, "maxiter": 1000},
        )

        if self.verbose:
            logger.info("Probabilistic meta-model fit: CRPS=%.4f, success=%s", res.fun, res.success)

        coef = np.asarray(res.x, float)
        if (not res.success) or (coef.shape != (n,)) or (not np.all(np.isfinite(coef))):
            logger.warning("Probabilistic meta-model fit failed; falling back to uniform weights")
            coef = np.ones(n, dtype=float) / n
        else:
            coef = np.maximum(coef, 0.0)
            coef /= coef.sum() + 1e-12

        self.coef_ = coef
        return self

    def predict(self, X_samples: list[np.ndarray]) -> np.ndarray:
        if self.coef_ is None:
            raise ValueError("Meta-model not fitted")
        if len(X_samples) != len(self.coef_):
            raise ValueError(f"Expected {len(self.coef_)} sample arrays, got {len(X_samples)}")
        ens = _vincentize_samples(X_samples, self.coef_)
        return np.asarray(np.maximum(ens, 0.0), float)


__all__ = [
    "NonNegativeMetaModel",
    "ProbabilisticMetaModel",
    "crps_ensemble",
]
