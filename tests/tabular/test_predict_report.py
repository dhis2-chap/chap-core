"""Tests for the `chap tabular predict` HTML performance report."""

import pandas as pd
import pytest

from chap_core.tabular.predict import PredictionResult, run_prediction
from chap_core.tabular.predict_report import render_prediction_report


def test_classification_report_has_confusion_matrix_and_roc(classifier_joblib, classification_csv):
    result = run_prediction(classifier_joblib, classification_csv, target="target")
    html = render_prediction_report(result)

    assert "Confusion matrix" in html
    assert "ROC curve" in html
    assert "balanced_accuracy" in html
    assert html.count("data:image/png;base64,") == 1


def test_regression_report_has_predicted_vs_actual_scatter(regressor_joblib, regression_csv):
    result = run_prediction(regressor_joblib, regression_csv, target="target")
    html = render_prediction_report(result)

    assert "Predicted vs actual" in html
    assert "Confusion matrix" not in html
    assert "data:image/png;base64," in html


def test_report_without_probability_falls_back_to_metric_bars(classification_frame):
    frame = classification_frame.assign(prediction=classification_frame["target"])
    result = PredictionResult(
        frame=frame,
        task="classification",
        has_probability=False,
        performance={"precision": 0.8, "recall": 0.7, "f1": 0.75, "accuracy": 0.78},
        target_name="target",
    )

    html = render_prediction_report(result)
    assert "Per-class metrics" in html
    assert "ROC curve" not in html


def test_report_requires_target(classifier_joblib, classification_frame, tmp_path):
    features_only = tmp_path / "features.csv"
    classification_frame.drop(columns=["target"]).to_csv(features_only, index=False)
    result = run_prediction(classifier_joblib, features_only, target="target")

    with pytest.raises(ValueError, match="needs the target column"):
        render_prediction_report(result)
