"""
Classes for surrogate models in LIME pipeline
"""

from dataclasses import dataclass
from typing import List, Optional, Protocol, Sequence, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.tree import DecisionTreeRegressor


@dataclass
class SurrogateResult:
    feature_names: Sequence[str]
    weighting: np.ndarray

    def as_sorted(self) -> List[Tuple[str, float]]:
        pairs = list(zip(self.feature_names, self.weighting.tolist()))
        return sorted(pairs, key=lambda t: -abs(t[1]))
    

class SurrogateModel(Protocol):
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "SurrogateModel": ...

    def explain(self, feature_names: List[str]) -> SurrogateResult: ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...





class RidgeSurrogate:
    def __init__(self, alpha: float = 5.0, fit_intercept: bool = True):
        self.alpha = alpha
        self.scaler: Optional[StandardScaler] = None
        self.fit_intercept = fit_intercept
        self.model: Optional[Ridge] = Ridge(alpha=self.alpha, fit_intercept=self.fit_intercept)

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> "RidgeSurrogate":
        self.model.fit(X, y, sample_weight=sample_weight)
        return self

    def explain(self, feature_names: List[str]) -> SurrogateResult:
        if self.model is None:
            raise RuntimeError("Surrogate not fitted.")
        return SurrogateResult(feature_names=feature_names, weighting=self.model.coef_.copy())

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Surrogate not fitted.")
        return self.model.predict(X)
    

class BayesianSurrogate:
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

    def fit(self, X, y, sample_weights):
        # Center data TODO: Should have fit_intercept arg?
        self.X_offset = np.average(X, axis=0, weights=sample_weights)
        self.y_offset = np.average(y, weights=sample_weights)
        Xc = X - self.X_offset
        yc = y - self.y_offset

        # From equation 3 in paper
        A = self.prior_prec * np.eye(X.shape[1])  # Prior information (np.eye gives identity matrix)
        A += self.noise_prec * (Xc.T @ (sample_weights[:, None] * Xc))  # Add information from weighted data points, with noise caveat

        sigma = np.linalg.pinv(A)  # Covariance matrix

        m = self.noise_prec * (sigma @ (Xc.T @ (sample_weights * yc)))  # Mean weights
        self.coef_ = m

        variances = np.diag(sigma)
        variances = np.clip(variances, 0.0, None)  # In case floating point math results in tiny negative number
        self.coef_std_ = np.sqrt(variances)  # Standard deviation is the sqrt of the variance, which you get from the diagonal of the covmat
        

        self.uncertainty = sigma
        self.weights = m
        self.intercept_ = self.y_offset - (self.X_offset @ self.coef_)

        return self
    
    def explain(self, feature_names):
        if self.coef_ is None:
            raise RuntimeError("Surrogate not fitted")
        return SurrogateResult(feature_names=feature_names, weighting=self.coef_.copy())

    def predict(self, X):
        if self.coef_ is None:
            raise RuntimeError("Surrogate not fitted")
        return X @ self.coef_ + self.intercept_
    
    def acquisition_scores(self, X_candidates, locality_weights):
        if self.uncertainty is None:
            raise RuntimeError("Surrogate not fitted")
        
        Xc = X_candidates - self.X_offset

        quad = np.einsum("ij,jk,ik->i", Xc, self.uncertainty, Xc)  # Fast einstein summation
        return locality_weights * quad