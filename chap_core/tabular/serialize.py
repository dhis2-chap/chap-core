"""Persist and restore a fitted phase-1 model as joblib or ONNX.

joblib keeps the native scikit-learn object. ONNX is an inference-only format
(scoring, and class probabilities for the classifier) and needs the optional
``onnx`` extra: ``pip install 'chap-core[onnx]'``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

import numpy as np

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator

ModelFormat = Literal["joblib", "onnx"]
Task = Literal["classification", "regression"]

_JOBLIB_SUFFIXES = {".joblib", ".pkl", ".pickle"}


class PredictModel(Protocol):
    """Minimal inference interface shared by the joblib and ONNX adapters."""

    task: Task

    def predict(self, features: np.ndarray) -> np.ndarray: ...

    def predict_proba(self, features: np.ndarray) -> np.ndarray | None:
        """Positive-class probabilities, or ``None`` when unavailable."""
        ...


def ensure_format_available(model_format: ModelFormat) -> None:
    """Fail fast (before a long evaluation) if the format's dependency is missing."""
    if model_format == "onnx":
        _import_skl2onnx()


def save_model(estimator: BaseEstimator, path: str | Path, model_format: ModelFormat) -> None:
    if model_format == "joblib":
        import joblib

        joblib.dump(estimator, path)
    elif model_format == "onnx":
        Path(path).write_bytes(_to_onnx_bytes(estimator))
    else:
        raise ValueError(f"Unknown model format {model_format!r}. Choose from: joblib, onnx")


def format_from_path(path: str | Path) -> ModelFormat:
    """Infer the model format from a file extension."""
    suffix = Path(path).suffix.lower()
    if suffix in _JOBLIB_SUFFIXES:
        return "joblib"
    if suffix == ".onnx":
        return "onnx"
    raise ValueError(f"Cannot infer model format from {Path(path).name!r}; expected .joblib or .onnx")


def load_model(path: str | Path) -> PredictModel:
    """Load a model saved by :func:`save_model`, format inferred from the extension."""
    if format_from_path(path) == "joblib":
        import joblib

        return _SklearnModel(joblib.load(path))
    return _OnnxModel(path)


class _SklearnModel:
    def __init__(self, estimator: BaseEstimator) -> None:
        from sklearn.base import is_classifier

        self._estimator = estimator
        self.task: Task = "classification" if is_classifier(estimator) else "regression"

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(self._estimator.predict(features))

    def predict_proba(self, features: np.ndarray) -> np.ndarray | None:
        if self.task != "classification" or not hasattr(self._estimator, "predict_proba"):
            return None
        return np.asarray(self._estimator.predict_proba(features))[:, 1]


class _OnnxModel:
    def __init__(self, path: str | Path) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "Loading an ONNX model requires the 'onnx' extra. Install it with: pip install 'chap-core[onnx]'"
            ) from exc

        self._session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self._input = self._session.get_inputs()[0].name
        outputs = [out.name for out in self._session.get_outputs()]
        # skl2onnx names a classifier's outputs *_label / *_probability and a
        # regressor's single output "variable".
        self._proba_output = next((name for name in outputs if "prob" in name), None)
        label_output = next((name for name in outputs if "label" in name), None)
        self.task: Task = "classification" if (label_output or self._proba_output) else "regression"
        self._label_output = label_output or outputs[0]

    def predict(self, features: np.ndarray) -> np.ndarray:
        raw = self._session.run([self._label_output], {self._input: features.astype(np.float32)})[0]
        return np.asarray(raw).ravel()

    def predict_proba(self, features: np.ndarray) -> np.ndarray | None:
        if self._proba_output is None:
            return None
        raw = self._session.run([self._proba_output], {self._input: features.astype(np.float32)})[0]
        rows = list(raw)
        if rows and isinstance(rows[0], dict):  # skl2onnx ZipMap output: one {class: prob} per row
            return np.array([row[max(row)] for row in rows], dtype=float)
        array = np.asarray(raw, dtype=float)
        return array[:, -1] if array.ndim == 2 else array


def _import_skl2onnx():
    try:
        from skl2onnx import to_onnx
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError as exc:
        raise RuntimeError(
            "ONNX export requires the 'onnx' extra. Install it with: pip install 'chap-core[onnx]'"
        ) from exc
    return to_onnx, FloatTensorType


def _to_onnx_bytes(estimator: BaseEstimator) -> bytes:
    to_onnx, float_tensor_type = _import_skl2onnx()
    n_features = int(estimator.n_features_in_)
    initial_types = [("input", float_tensor_type([None, n_features]))]
    return bytes(to_onnx(estimator, initial_types=initial_types).SerializeToString())
