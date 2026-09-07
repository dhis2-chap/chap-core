"""Persist a fitted phase-1 model as joblib or ONNX.

joblib keeps the native scikit-learn object. ONNX is an inference-only export
(scoring, and class probabilities for the classifier) and needs the optional
``onnx`` extra: ``pip install 'chap-core[onnx]'``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator

ModelFormat = Literal["joblib", "onnx"]


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
