


from dataclasses import dataclass
from typing import List, Optional, Protocol, Sequence, Tuple

import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor


@dataclass
class SurrogateResult:
    feature_names: Sequence[str]
    weighting: np.ndarray

    def as_sorted(self) -> List[Tuple[str, float]]:
        pairs = list(zip(self.feature_names, self.weighting.tolist()))
        return sorted(pairs, key=lambda t: -abs(t[1]))
    

class SurrogateModel(Protocol):
    """
    Minimal contract:
      - fit on X, y with optional kernel weights
      - expose local importances aligned with feature_names
      - optionally predict (useful for sanity checks)
    """
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "SurrogateModel": ...

    def explain(self, feature_names: List[str]) -> SurrogateResult: ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...





class RidgeSurrogate:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.scaler: Optional[StandardScaler] = None
        self.model: Optional[Ridge] = None

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> "RidgeSurrogate":
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X)

        self.model = Ridge(alpha=self.alpha)
        self.model.fit(Xs, y, sample_weight=sample_weight)
        return self

    def explain(self, feature_names: List[str]) -> SurrogateResult:
        if self.model is None:
            raise RuntimeError("Surrogate not fitted.")
        return SurrogateResult(feature_names=feature_names, weighting=self.model.coef_.copy())

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None or self.scaler is None:
            raise RuntimeError("Surrogate not fitted.")
        return self.model.predict(self.scaler.transform(X))
    

class TreeSurrogate:
    def __init__(self, max_depth: int = 3, random_state: int = 0):
        self.max_depth = max_depth
        self.random_state = random_state
        self.model: Optional[DecisionTreeRegressor] = None

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> "TreeSurrogate":
        self.model = DecisionTreeRegressor(max_depth=self.max_depth, random_state=self.random_state)
        # DecisionTreeRegressor supports sample_weight
        self.model.fit(X, y, sample_weight=sample_weight)
        return self

    def explain(self, feature_names: List[str]) -> SurrogateResult:
        if self.model is None:
            raise RuntimeError("Surrogate not fitted.")
        return SurrogateResult(feature_names=feature_names, weighting=self.model.feature_importances_.copy())

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Surrogate not fitted.")
        return self.model.predict(X)