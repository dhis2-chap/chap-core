# tabular Command Reference

The `chap tabular` commands evaluate plain (non-temporal) tabular models and run
predictions from a saved model. Phase 1 supports two models: `logistic_regression`
for binary classification and `ridge` for regression.

The input CSV must be fully preprocessed: numeric or encoded columns, no missing
values, no exact duplicate rows, and one target column. These assumptions are
checked - the command refuses to run and names the offending columns.

## Synopsis

```console
chap tabular evaluate <MODEL> <DATASET_CSV> [OPTIONS]
chap tabular predict  <MODEL_PATH> <DATASET_CSV> [OPTIONS]
```

## `chap tabular evaluate`

Runs seeded 5-fold cross-validation (stratified for classification) and writes
`results.json` (per-fold and mean +/- std metrics) and `report.html`.

With `--model-output` it additionally holds out a train/test split: cross-validation
runs on the training split, a final model is trained on it and scored once on the
held-out test split, and that model is saved. `results.json` and the report then
carry both the cross-validation and the test numbers.

| Option | Description | Default |
|--------|-------------|---------|
| `--output-folder` | Directory for all result files (created if missing) | `tabular_results` |
| `--output` | Results JSON filename | `results.json` |
| `--report` | HTML report filename | `report.html` |
| `--model-output` | If set, save the trained model here (enables the held-out split) | none |
| `--model-format` | `joblib` or `onnx` (`onnx` needs `pip install 'chap-core[onnx]'`) | `joblib` |
| `--target` | Name of the target column in the CSV | `target` |

### Example

Train `logistic_regression` on the 70% diabetes split, save the model, and write
the HTML report:

```console
chap tabular evaluate logistic_regression example_data/diabetes_tabular_train_70.csv \
    --target Outcome \
    --model-output diabetes_model.joblib \
    --output-folder tabular_run
```

This writes `tabular_run/results.json`, `tabular_run/report.html`, and
`tabular_run/diabetes_model.joblib`.

## `chap tabular predict`

Loads a saved model (format inferred from the extension: `.joblib` or `.onnx`),
scores every row, and writes `predictions.csv` - the input rows with a
`prediction` column appended, plus a `probability` column for a classifier that
exposes one.

If the dataset contains the target column, it also writes
`predictions.performance.json` and `predictions.report.html`. The HTML report
shows the metrics, a confusion matrix and a ROC curve for classification, or a
predicted vs actual scatter for regression. These are external test numbers, not
cross-validation.

| Option | Description | Default |
|--------|-------------|---------|
| `--output-folder` | Directory for all result files (created if missing) | `tabular_predictions` |
| `--output` | Predictions CSV filename | `predictions.csv` |
| `--target` | Target column name; if present in the dataset, the reports are written | `target` |

### Example

Score the held-out 30% diabetes split with the model trained above:

```console
chap tabular predict tabular_run/diabetes_model.joblib example_data/diabetes_tabular_test_30.csv \
    --target Outcome \
    --output-folder tabular_predictions
```

This writes `tabular_predictions/predictions.csv`,
`tabular_predictions/predictions.performance.json`, and
`tabular_predictions/predictions.report.html`.

## See Also

- [eval Reference](eval-reference.md) - rolling-origin backtest evaluation for temporal models
