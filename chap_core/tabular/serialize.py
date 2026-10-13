"""Persist and restore a fitted tabular model with joblib."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path

    from sklearn.base import BaseEstimator

Task = Literal["classification", "regression"]


class PredictModel(Protocol):
    """Minimal inference interface implemented by the loaded model."""

    task: Task
    feature_names: list[str] | None
    """Training column names in order, or ``None`` if the model does not record them."""

    def predict(self, features: np.ndarray) -> np.ndarray: ...

    def predict_proba(self, features: np.ndarray) -> np.ndarray | None:
        """Positive-class probabilities, or ``None`` when unavailable."""
        ...


def save_model(estimator: BaseEstimator, path: str | Path) -> None:
    import joblib

    joblib.dump(estimator, path)


def load_model(path: str | Path) -> PredictModel:
    """Load a model saved by :func:`save_model`.

    joblib files are pickles and can execute arbitrary code on load: only use
    this on local, trusted paths and never expose it through the REST API.
    """
    import joblib

    return _SklearnModel(joblib.load(path))


class _SklearnModel:
    def __init__(self, estimator: BaseEstimator) -> None:
        from sklearn.base import is_classifier

        self._estimator = estimator
        self.task: Task = "classification" if is_classifier(estimator) else "regression"
        names = getattr(estimator, "feature_names_in_", None)
        self.feature_names: list[str] | None = None if names is None else [str(n) for n in names]

    def _frame(self, features: np.ndarray) -> np.ndarray | pd.DataFrame:
        return features if self.feature_names is None else pd.DataFrame(features, columns=self.feature_names)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(self._estimator.predict(self._frame(features)))

    def predict_proba(self, features: np.ndarray) -> np.ndarray | None:
        if self.task != "classification" or not hasattr(self._estimator, "predict_proba"):
            return None
        return np.asarray(self._estimator.predict_proba(self._frame(features)))[:, 1]
