"""The two phase-1 models: logistic regression and ridge.

Nothing else is selectable. Each entry fixes the task type (classification or
regression) and the estimator; the CLI ``MODEL`` argument is restricted to
these keys.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from sklearn.linear_model import LogisticRegression, Ridge

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator

TaskType = Literal["classification", "regression"]


@dataclass(frozen=True)
class TabularModel:
    name: str
    task: TaskType

    def build(self) -> BaseEstimator:
        if self.name == "logistic_regression":
            return LogisticRegression(max_iter=1000)
        return Ridge()


_MODELS: dict[str, TabularModel] = {
    "logistic_regression": TabularModel("logistic_regression", "classification"),
    "ridge": TabularModel("ridge", "regression"),
}

MODEL_NAMES = tuple(_MODELS)


def get_model(name: str) -> TabularModel:
    """Return the :class:`TabularModel` for ``name`` or raise ``ValueError``."""
    try:
        return _MODELS[name]
    except KeyError:
        raise ValueError(f"Unknown model {name!r}. Choose from: {list(MODEL_NAMES)}") from None
