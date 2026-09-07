"""Tests for persisting a fitted phase-1 model."""

import sys

import numpy as np
import pytest

from chap_core.tabular.cv import evaluate_tabular
from chap_core.tabular.dataset import load_tabular_dataset
from chap_core.tabular.model import get_model
from chap_core.tabular.serialize import save_model


@pytest.fixture
def fitted_estimator(regression_csv):
    _, estimator = evaluate_tabular(load_tabular_dataset(regression_csv), get_model("ridge"), holdout=True)
    return estimator


def test_save_joblib_roundtrips(fitted_estimator, regression_frame, tmp_path):
    joblib = pytest.importorskip("joblib")
    path = tmp_path / "model.joblib"

    save_model(fitted_estimator, path, "joblib")

    features = regression_frame[["x1", "x2"]].to_numpy()
    reloaded = joblib.load(path)
    np.testing.assert_allclose(reloaded.predict(features), fitted_estimator.predict(features))


def test_unknown_format_is_rejected(fitted_estimator, tmp_path):
    with pytest.raises(ValueError, match="Unknown model format"):
        save_model(fitted_estimator, tmp_path / "model.bin", "pickle")


def test_onnx_without_extra_raises_with_install_hint(fitted_estimator, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "skl2onnx", None)  # force ImportError on `import skl2onnx`
    with pytest.raises(RuntimeError, match=r"chap-core\[onnx\]"):
        save_model(fitted_estimator, tmp_path / "model.onnx", "onnx")


def test_onnx_export_writes_valid_graph(fitted_estimator, tmp_path):
    pytest.importorskip("skl2onnx")
    onnx = pytest.importorskip("onnx")
    path = tmp_path / "model.onnx"

    save_model(fitted_estimator, path, "onnx")

    graph = onnx.load(str(path)).graph
    assert len(graph.node) > 0
    assert graph.input[0].type.tensor_type.shape.dim[1].dim_value == 2
