"""Tests for seeded cross-validation of the tabular models."""

from chap_core.tabular.cv import N_SPLITS, cross_validate, evaluate_tabular
from chap_core.tabular.dataset import load_tabular_dataset
from chap_core.tabular.model import get_model


def test_classification_cv_produces_fixed_metric_set(classification_csv):
    dataset = load_tabular_dataset(classification_csv)
    result = cross_validate(dataset, get_model("logistic_regression"))

    assert result.task == "classification"
    assert len(result.per_fold) == N_SPLITS
    assert set(result.mean) == {
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
    }
    assert result.headline_metric == "balanced_accuracy"
    assert 0.0 <= result.mean["balanced_accuracy"] <= 1.0


def test_regression_cv_produces_fixed_metric_set(regression_csv):
    dataset = load_tabular_dataset(regression_csv)
    result = cross_validate(dataset, get_model("ridge"))

    assert result.task == "regression"
    assert set(result.mean) == {"mae", "rmse", "r2"}
    assert result.headline_metric == "mae"
    assert result.mean["r2"] > 0.5


def test_cv_is_deterministic(classification_csv):
    dataset = load_tabular_dataset(classification_csv)
    model = get_model("logistic_regression")
    first = cross_validate(dataset, model).to_dict()
    second = cross_validate(dataset, model).to_dict()
    assert first == second


def test_result_dict_reports_cross_validation(regression_csv):
    dataset = load_tabular_dataset(regression_csv)
    result = cross_validate(dataset, get_model("ridge")).to_dict()
    assert result["cross_validation"] is True
    assert result["n_splits"] == N_SPLITS
    assert len(result["per_fold"]) == N_SPLITS
    assert set(result["std"]) == set(result["mean"])
    assert "test" not in result


def test_evaluate_tabular_without_holdout_matches_plain_cv(classification_csv):
    dataset = load_tabular_dataset(classification_csv)
    model = get_model("logistic_regression")
    result, estimator = evaluate_tabular(dataset, model, holdout=False)
    assert estimator is None
    assert result.test is None
    assert result.to_dict() == cross_validate(dataset, model).to_dict()


def test_evaluate_tabular_with_holdout_reports_cv_and_test(classification_csv):
    dataset = load_tabular_dataset(classification_csv)
    result, estimator = evaluate_tabular(dataset, get_model("logistic_regression"), holdout=True)

    assert estimator is not None and hasattr(estimator, "predict")
    assert result.test is not None
    assert result.n_samples == result.test["n_train"]
    assert result.n_samples + result.test["n_test"] == len(dataset)
    assert set(result.test["metrics"]) == set(result.mean)

    payload = result.to_dict()
    assert payload["test"]["metrics"][payload["headline_metric"]] == result.test["metrics"]["balanced_accuracy"]


def test_evaluate_tabular_holdout_is_deterministic(regression_csv):
    dataset = load_tabular_dataset(regression_csv)
    model = get_model("ridge")
    first, _ = evaluate_tabular(dataset, model, holdout=True)
    second, _ = evaluate_tabular(dataset, model, holdout=True)
    assert first.to_dict() == second.to_dict()
