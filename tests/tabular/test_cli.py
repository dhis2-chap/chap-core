"""Tests for the `chap tabular evaluate` CLI command."""

import json
from pathlib import Path

import pytest

from chap_core.cli_endpoints.tabular import evaluate
from chap_core.tabular.dataset import DatasetAssumptionError


def test_evaluate_writes_all_files_into_output_folder(classification_csv, tmp_path):
    folder = tmp_path / "runs"

    evaluate("logistic_regression", classification_csv, output_folder=folder)

    results = json.loads((folder / "results.json").read_text())
    assert results["model"] == "logistic_regression"
    assert results["cross_validation"] is True
    assert len(results["per_fold"]) == 5

    html = (folder / "report.html").read_text()
    assert "cross-validation numbers" in html
    assert "balanced_accuracy" in html


def test_evaluate_creates_missing_output_folder(regression_csv, tmp_path):
    folder = tmp_path / "a" / "b" / "c"
    evaluate("ridge", regression_csv, output_folder=folder)
    assert json.loads((folder / "results.json").read_text())["task"] == "regression"


def test_evaluate_absolute_output_path_overrides_folder(classification_csv, tmp_path):
    folder = tmp_path / "runs"
    explicit = tmp_path / "elsewhere" / "custom.json"
    explicit.parent.mkdir()

    evaluate("logistic_regression", classification_csv, output_folder=folder, output=explicit)

    assert explicit.exists()
    assert not (folder / "results.json").exists()


def test_evaluate_with_model_output_saves_model_and_test_metrics(classification_csv, tmp_path):
    import joblib

    folder = tmp_path / "runs"
    evaluate(
        "logistic_regression",
        classification_csv,
        output_folder=folder,
        model_output=Path("model.joblib"),
    )

    estimator = joblib.load(folder / "model.joblib")
    assert hasattr(estimator, "predict")

    results = json.loads((folder / "results.json").read_text())
    assert results["cross_validation"] is True
    assert set(results["test"]["metrics"]) == set(results["mean"])
    assert results["test"]["n_train"] + results["test"]["n_test"] == 60

    html = (folder / "report.html").read_text()
    assert "Held-out test metrics" in html
    assert "not cross-validation" in html


def test_evaluate_model_format_onnx(classification_csv, tmp_path):
    pytest.importorskip("skl2onnx")
    folder = tmp_path / "runs"

    evaluate(
        "logistic_regression",
        classification_csv,
        output_folder=folder,
        model_output=Path("model.onnx"),
        model_format="onnx",
    )

    assert (folder / "model.onnx").stat().st_size > 0
    assert "test" in json.loads((folder / "results.json").read_text())


def test_evaluate_without_model_output_has_no_test_section(regression_csv, tmp_path):
    folder = tmp_path / "runs"
    evaluate("ridge", regression_csv, output_folder=folder)
    assert "test" not in json.loads((folder / "results.json").read_text())
    assert not (folder / "model.joblib").exists()


def test_evaluate_honours_target_option(regression_frame, tmp_path):
    regression_frame = regression_frame.rename(columns={"target": "y"})
    path = tmp_path / "renamed.csv"
    regression_frame.to_csv(path, index=False)

    evaluate("ridge", path, output_folder=tmp_path / "runs", target="y")
    assert json.loads((tmp_path / "runs" / "results.json").read_text())["target"] == "y"


def test_evaluate_rejects_bad_dataset(classification_frame, tmp_path):
    classification_frame.loc[0, "x1"] = None
    path = tmp_path / "bad.csv"
    classification_frame.to_csv(path, index=False)

    with pytest.raises(DatasetAssumptionError):
        evaluate("logistic_regression", path, output_folder=tmp_path / "runs")
