"""Tests for the saved-model prediction runner."""

import pandas as pd
import pytest

from chap_core.tabular.dataset import DatasetAssumptionError
from chap_core.tabular.predict import run_prediction


def _drop_target(csv_path, tmp_path):
    frame = pd.read_csv(csv_path).drop(columns=["target"])
    path = tmp_path / "no_target.csv"
    frame.to_csv(path, index=False)
    return path


def test_classifier_appends_class_and_probability(classifier_joblib, classification_csv):
    result = run_prediction(classifier_joblib, classification_csv)

    assert result.task == "classification"
    assert result.has_probability
    assert list(result.frame.columns[-2:]) == ["prediction", "probability"]
    assert len(result.frame) == 60
    assert set(result.frame["prediction"].unique()) <= {0, 1}
    assert result.frame["probability"].between(0.0, 1.0).all()


def test_regressor_appends_prediction_only(regressor_joblib, regression_csv):
    result = run_prediction(regressor_joblib, regression_csv)

    assert result.task == "regression"
    assert not result.has_probability
    assert result.frame.columns[-1] == "prediction"
    assert "probability" not in result.frame.columns


def test_performance_report_only_when_target_present(classifier_joblib, classification_csv, tmp_path):
    with_target = run_prediction(classifier_joblib, classification_csv)
    assert with_target.performance is not None
    assert set(with_target.performance) == {
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
    }

    without_target = run_prediction(classifier_joblib, _drop_target(classification_csv, tmp_path))
    assert without_target.performance is None
    assert "prediction" in without_target.frame.columns


def test_regression_performance_metrics(regressor_joblib, regression_csv):
    result = run_prediction(regressor_joblib, regression_csv)
    assert set(result.performance) == {"mae", "rmse", "r2"}


def test_missing_feature_values_are_rejected(classifier_joblib, classification_frame, tmp_path):
    classification_frame.loc[0, "x1"] = None
    path = tmp_path / "bad.csv"
    classification_frame.to_csv(path, index=False)

    with pytest.raises(DatasetAssumptionError, match="missing values"):
        run_prediction(classifier_joblib, path)


def test_onnx_classifier_predictions_match_joblib(classifier_onnx, classifier_joblib, classification_csv, tmp_path):
    features_only = _drop_target(classification_csv, tmp_path)

    onnx_result = run_prediction(classifier_onnx, features_only)
    joblib_result = run_prediction(classifier_joblib, features_only)

    assert onnx_result.task == "classification"
    assert (onnx_result.frame["prediction"].to_numpy() == joblib_result.frame["prediction"].to_numpy()).all()
    assert onnx_result.frame["probability"].between(0.0, 1.0).all()
