"""Tabular model evaluation commands for the CHAP CLI."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated, Literal

from cyclopts import Parameter

logger = logging.getLogger(__name__)


def evaluate(
    model: Annotated[
        Literal["logistic_regression", "ridge"],
        Parameter(help="Model to evaluate: logistic_regression (classification) or ridge (regression)."),
    ],
    dataset_csv: Annotated[Path, Parameter(help="Path to a fully preprocessed CSV dataset.")],
    *,
    output_folder: Annotated[
        Path,
        Parameter(help="Directory to write result files into (created if missing)."),
    ] = Path("tabular_results"),
    output: Annotated[
        Path,
        Parameter(help="Results JSON filename; a relative path is placed inside --output-folder."),
    ] = Path("results.json"),
    report: Annotated[
        Path,
        Parameter(help="HTML report filename; a relative path is placed inside --output-folder."),
    ] = Path("report.html"),
    model_output: Annotated[
        Path | None,
        Parameter(
            help="If set, hold out a test split, train a final model on the training split, "
            "score it on the test split, and save it here. Relative paths go inside "
            "--output-folder."
        ),
    ] = None,
    model_format: Annotated[
        Literal["joblib", "onnx"],
        Parameter(help="Format for --model-output. 'onnx' needs the 'onnx' extra: pip install 'chap-core[onnx]'."),
    ] = "joblib",
    target: Annotated[str, Parameter(help="Name of the target column in the CSV.")] = "target",
):
    """Evaluate a tabular model with seeded 5-fold cross-validation.

    Reads an already preprocessed dataset (numeric or encoded, no missing
    values, deduplicated, one target), runs stratified 5-fold CV for
    classification or plain 5-fold CV for regression, and writes per-fold and
    mean +/- std metrics plus an HTML report.

    Without ``--model-output`` there is no held-out test set and every number is
    a cross-validation number. With ``--model-output`` the data is split into a
    training and a held-out test set: cross-validation runs on the training
    split, a final model is trained on that split and scored once on the test
    split, and that model is saved. The report and results.json then carry both
    the cross-validation and the test numbers.

    All result files are written into ``--output-folder`` (default
    ``tabular_results/``) unless the filename options are given as absolute
    paths.

    Input assumptions are checked, not assumed: the command refuses to run,
    naming the offending columns, on missing values, non-numeric unencoded
    columns, exact duplicate rows, or a constant target.

    Examples:
        chap tabular evaluate logistic_regression ./data.csv
        chap tabular evaluate ridge ./data.csv --output-folder ./runs/ridge
        chap tabular evaluate logistic_regression ./data.csv --model-output model.joblib
        chap tabular evaluate ridge ./data.csv --model-output model.onnx --model-format onnx
    """
    from chap_core.tabular.cv import evaluate_tabular
    from chap_core.tabular.dataset import load_tabular_dataset
    from chap_core.tabular.model import get_model
    from chap_core.tabular.report import write_report
    from chap_core.tabular.serialize import ensure_format_available, save_model

    if model_output is not None:
        ensure_format_available(model_format)

    dataset = load_tabular_dataset(dataset_csv, target=target)
    result, estimator = evaluate_tabular(dataset, get_model(model), holdout=model_output is not None)

    output_folder.mkdir(parents=True, exist_ok=True)

    def _resolve(path: Path) -> Path:
        return path if path.is_absolute() else output_folder / path

    results_path, report_path = _resolve(output), _resolve(report)

    results_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    logger.info(f"Wrote results to {results_path}")

    write_report(result, report_path)
    logger.info(f"Wrote report to {report_path}")

    headline = result.headline_metric
    logger.info(f"{headline} (CV): {result.mean[headline]:.4f} +/- {result.std[headline]:.4f}")

    if estimator is not None and model_output is not None and result.test is not None:
        model_path = _resolve(model_output)
        save_model(estimator, model_path, model_format)
        logger.info(f"Wrote trained model ({model_format}) to {model_path}")
        logger.info(f"{headline} (test): {result.test['metrics'][headline]:.4f}")


def predict(
    model_path: Annotated[Path, Parameter(help="Path to a saved model (.joblib or .onnx).")],
    dataset_csv: Annotated[Path, Parameter(help="Path to the prediction dataset CSV.")],
    *,
    output_folder: Annotated[
        Path,
        Parameter(help="Directory to write result files into (created if missing)."),
    ] = Path("tabular_predictions"),
    output: Annotated[
        Path,
        Parameter(help="Predictions CSV filename; a relative path is placed inside --output-folder."),
    ] = Path("predictions.csv"),
    target: Annotated[
        str, Parameter(help="Target column name; if present in the dataset, a report is written.")
    ] = "target",
):
    """Score a dataset with a saved tabular model.

    Loads the model (format inferred from the extension: .joblib or .onnx),
    predicts every row, and writes the input rows with a ``prediction`` column
    appended - plus a ``probability`` column for a classifier that exposes one.

    If the dataset contains the ``--target`` column, a ``<output>.performance.json``
    and an HTML report (``<output>.report.html``) are also written, using the
    same fixed metric sets as ``evaluate`` (classification: accuracy, balanced
    accuracy, precision/recall/F1, ROC-AUC, PR-AUC; regression: MAE, RMSE, R2).
    The report adds a confusion matrix and a ROC curve for classification, or a
    predicted vs actual scatter for regression. These are external test numbers,
    not cross-validation.

    Feature columns must be numeric or encoded with no missing values.

    Examples:
        chap tabular predict model.joblib ./new_data.csv
        chap tabular predict model.onnx ./new_data.csv --output-folder ./runs
    """
    from chap_core.tabular.predict import run_prediction
    from chap_core.tabular.predict_report import write_prediction_report

    result = run_prediction(model_path, dataset_csv, target=target)

    output_folder.mkdir(parents=True, exist_ok=True)
    predictions_path = output if output.is_absolute() else output_folder / output
    result.frame.to_csv(predictions_path, index=False)
    logger.info(f"Wrote {len(result.frame)} predictions to {predictions_path}")

    if result.performance is not None:
        performance_path = predictions_path.with_name(predictions_path.stem + ".performance.json")
        performance_path.write_text(json.dumps(result.performance, indent=2), encoding="utf-8")
        logger.info(f"Wrote performance report to {performance_path}")

        report_path = predictions_path.with_name(predictions_path.stem + ".report.html")
        write_prediction_report(result, report_path)
        logger.info(f"Wrote HTML report to {report_path}")
    else:
        logger.info(f"No {target!r} column in the dataset; skipping the performance report")


def register_commands(app):
    from cyclopts import App

    tabular_app = App(name="tabular", help="Tabular model evaluation.")
    tabular_app.command(name="evaluate")(evaluate)
    tabular_app.command(name="predict")(predict)
    app.command(tabular_app)
