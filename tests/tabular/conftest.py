"""Fixtures for phase-1 tabular evaluation tests.

The existing conftest fixtures are all spatio-temporal disease data, which the
tabular subsystem does not consume, so small preprocessed tables are defined
here.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def classification_frame():
    """A balanced, fully preprocessed binary-classification table."""
    rng = np.random.default_rng(0)
    n = 60
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    logits = 1.5 * x1 - x2
    target = (logits + rng.normal(scale=0.5, size=n) > 0).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "target": target})


@pytest.fixture
def regression_frame():
    """A fully preprocessed regression table."""
    rng = np.random.default_rng(1)
    n = 60
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    target = 2.0 * x1 - 0.5 * x2 + rng.normal(scale=0.3, size=n)
    return pd.DataFrame({"x1": x1, "x2": x2, "target": target})


@pytest.fixture
def classification_csv(classification_frame, tmp_path):
    path = tmp_path / "classification.csv"
    classification_frame.to_csv(path, index=False)
    return path


@pytest.fixture
def regression_csv(regression_frame, tmp_path):
    path = tmp_path / "regression.csv"
    regression_frame.to_csv(path, index=False)
    return path


def _train(csv_path, model_name):
    from chap_core.tabular.cv import evaluate_tabular
    from chap_core.tabular.dataset import load_tabular_dataset
    from chap_core.tabular.model import get_model

    _, estimator = evaluate_tabular(load_tabular_dataset(csv_path), get_model(model_name), holdout=True)
    return estimator


@pytest.fixture
def classifier_joblib(classification_csv, tmp_path):
    from chap_core.tabular.serialize import save_model

    path = tmp_path / "classifier.joblib"
    save_model(_train(classification_csv, "logistic_regression"), path, "joblib")
    return path


@pytest.fixture
def regressor_joblib(regression_csv, tmp_path):
    from chap_core.tabular.serialize import save_model

    path = tmp_path / "regressor.joblib"
    save_model(_train(regression_csv, "ridge"), path, "joblib")
    return path


@pytest.fixture
def classifier_onnx(classification_csv, tmp_path):
    pytest.importorskip("skl2onnx")
    pytest.importorskip("onnxruntime")
    from chap_core.tabular.serialize import save_model

    path = tmp_path / "classifier.onnx"
    save_model(_train(classification_csv, "logistic_regression"), path, "onnx")
    return path
