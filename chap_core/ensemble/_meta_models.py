"""Meta-models and CRPS utilities for stacking ensembles."""

from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import minimize, nnls

logger = logging.getLogger(__name__)


def _crps_score(obs: np.ndarray, forecast: np.ndarray) -> float:
    obs = np.asarray(obs, float).reshape(-1)
    forecast = np.asarray(forecast, float)
    if forecast.ndim != 2:
        raise ValueError(f"forecast must be 2D (n, m), got shape {forecast.shape}")
    n, m = forecast.shape
    if n != obs.shape[0]:
        raise ValueError(f"obs length {obs.shape[0]} does not match forecast rows {n}")

    term1 = np.mean(np.abs(forecast - obs[:, None]), axis=1)  # shape (n,)

    if m <= 1:
        return float(np.mean(term1))

    sorted_f = np.sort(forecast, axis=1)
    cumsum_f = np.cumsum(sorted_f, axis=1)

    k = np.arange(m)

    left = sorted_f * k - cumsum_f + sorted_f

    rev_cumsum_f = np.cumsum(sorted_f[:, ::-1], axis=1)[:, ::-1]
    right = rev_cumsum_f - sorted_f * (m - k)

    pairwise = left + right  # shape (n, m)
    sum_pairwise = 0.5 * np.sum(pairwise, axis=1)

    denom = m * (m - 1) / 2.0
    term2 = sum_pairwise / denom

    return float(np.mean(term1 - 0.5 * term2))


def crps_ensemble(observations: np.ndarray, forecasts: np.ndarray) -> float:
    return _crps_score(observations, forecasts)


class NonNegativeMetaModel:
    def __init__(self) -> None:
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> NonNegativeMetaModel:
        coef_raw, _ = nnls(X, y)  # type: ignore[misc]
        coef = np.asarray(coef_raw, float)
        s = coef.sum()
        self.coef_ = coef / s if s > 0 else coef
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise ValueError("Meta-model not fitted")
        return np.asarray(np.dot(X, self.coef_), float)


class ProbabilisticMetaModel:
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
            # w is already constrained to simplex; no need to renormalize
            ens = sum(w[i] * X_samples[i] for i in range(len(X_samples)))
            return _crps_score(y, ens)

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

        n = len(X_samples)
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
        ens = sum(self.coef_[i] * X_samples[i] for i in range(len(X_samples)))
        return np.asarray(np.maximum(ens, 0.0), float)
