"""Tests for persisting and restoring a fitted tabular model."""

import numpy as np
import pytest

from chap_core.tabular.cv import evaluate_tabular
from chap_core.tabular.dataset import load_tabular_dataset
from chap_core.tabular.model import get_model
from chap_core.tabular.serialize import load_model, save_model


@pytest.fixture
def fitted_estimator(regression_csv):
    _, estimator = evaluate_tabular(load_tabular_dataset(regression_csv), get_model("ridge"), holdout=True)
    return estimator


def test_save_joblib_roundtrips(fitted_estimator, regression_frame, tmp_path):
    joblib = pytest.importorskip("joblib")
    path = tmp_path / "model.joblib"

    save_model(fitted_estimator, path)

    features = regression_frame[["x1", "x2"]]
    reloaded = joblib.load(path)
    np.testing.assert_allclose(reloaded.predict(features), fitted_estimator.predict(features))


def test_load_joblib_model_predicts(regressor_joblib, regression_frame):
    model = load_model(regressor_joblib)
    assert model.task == "regression"
    assert model.predict_proba(regression_frame[["x1", "x2"]].to_numpy()) is None
    assert len(model.predict(regression_frame[["x1", "x2"]].to_numpy())) == 60
