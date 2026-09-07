"""Apply a saved tabular model to a new dataset.

Loads a model (joblib or ONNX), scores every row, and appends the predictions
to the input frame. When the dataset carries the target column, the same fixed
metric sets used by ``evaluate`` are computed on it - these are external test
numbers, not cross-validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from chap_core.tabular.dataset import validate_feature_frame
from chap_core.tabular.metrics import classification_metrics, regression_metrics
from chap_core.tabular.serialize import load_model

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class PredictionResult:
    """The input frame with prediction columns appended, plus optional metrics."""

    frame: pd.DataFrame
    task: str
    has_probability: bool
    performance: dict[str, float] | None
    target_name: str | None = None


def run_prediction(model_path: str | Path, dataset_csv: str | Path, target: str = "target") -> PredictionResult:
    """Score ``dataset_csv`` with the model at ``model_path``.

    Appends a ``prediction`` column (and ``probability`` for a classifier that
    exposes one). If ``target`` is a column in the dataset, also returns a
    performance report computed against it.
    """
    model = load_model(model_path)
    frame = pd.read_csv(dataset_csv)

    has_target = target in frame.columns
    features = frame.drop(columns=[target]) if has_target else frame
    validate_feature_frame(features)
    matrix = features.to_numpy()

    predictions = model.predict(matrix)
    probabilities = model.predict_proba(matrix)

    out = frame.copy()
    out["prediction"] = predictions
    if model.task == "classification" and probabilities is not None:
        out["probability"] = probabilities

    performance: dict[str, float] | None = None
    if has_target:
        y_true = frame[target].to_numpy()
        if model.task == "classification":
            scores = probabilities if probabilities is not None else predictions
            raw = classification_metrics(y_true, np.asarray(scores, dtype=float))
        else:
            raw = regression_metrics(y_true, np.asarray(predictions, dtype=float))
        performance = {name: float(value) for name, value in raw.items()}

    return PredictionResult(
        frame=out,
        task=model.task,
        has_probability=model.task == "classification" and probabilities is not None,
        performance=performance,
        target_name=target if has_target else None,
    )
