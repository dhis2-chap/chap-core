"""Seeded 5-fold cross-validation for the tabular models.

Stratified for classification, plain K-fold for regression. By default there is
no held-out test set and every number is a cross-validation number. When a
trained model is requested the data is split into a train and a held-out test
set: cross-validation then runs on the training split and the saved model is
also scored once on the test split.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from chap_core.tabular.dataset import TabularDataset
from chap_core.tabular.metrics import (
    CLASSIFICATION_HEADLINE,
    REGRESSION_HEADLINE,
    classification_metrics,
    regression_metrics,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from sklearn.base import BaseEstimator

    from chap_core.tabular.model import TabularModel

SEED = 42
N_SPLITS = 5
TEST_SIZE = 0.2


@dataclass(frozen=True)
class EvaluationResult:
    model: str
    task: str
    target: str
    n_samples: int
    n_splits: int
    seed: int
    feature_names: list[str]
    headline_metric: str
    per_fold: list[dict[str, float]] = field(default_factory=list)
    mean: dict[str, float] = field(default_factory=dict)
    std: dict[str, float] = field(default_factory=dict)
    test: dict | None = None

    def to_dict(self) -> dict:
        payload = {
            "model": self.model,
            "task": self.task,
            "target": self.target,
            "n_samples": self.n_samples,
            "n_splits": self.n_splits,
            "seed": self.seed,
            "feature_names": self.feature_names,
            "headline_metric": self.headline_metric,
            "cross_validation": True,
            "per_fold": self.per_fold,
            "mean": self.mean,
            "std": self.std,
        }
        if self.test is not None:
            payload["test"] = self.test
        return payload


def _score_fn(task: str) -> Callable[[np.ndarray, np.ndarray], dict[str, float]]:
    return classification_metrics if task == "classification" else regression_metrics


def _predict(estimator: BaseEstimator, features: np.ndarray, task: str) -> np.ndarray:
    if task == "classification":
        return np.asarray(estimator.predict_proba(features))[:, 1]
    return np.asarray(estimator.predict(features))


def cross_validate(dataset: TabularDataset, model: TabularModel) -> EvaluationResult:
    """Run seeded 5-fold CV and return per-fold and aggregated metrics."""
    features = dataset.features.to_numpy()
    target = dataset.target.to_numpy()
    score = _score_fn(model.task)

    if model.task == "classification":
        splitter: KFold = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        headline = CLASSIFICATION_HEADLINE
    else:
        splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        headline = REGRESSION_HEADLINE

    per_fold: list[dict[str, float]] = []
    for train_idx, test_idx in splitter.split(features, target):
        estimator = model.build()
        estimator.fit(features[train_idx], target[train_idx])
        prediction = _predict(estimator, features[test_idx], model.task)
        per_fold.append({k: float(v) for k, v in score(target[test_idx], prediction).items()})

    metric_names = list(per_fold[0])
    mean = {name: float(np.mean([fold[name] for fold in per_fold])) for name in metric_names}
    std = {name: float(np.std([fold[name] for fold in per_fold], ddof=1)) for name in metric_names}

    return EvaluationResult(
        model=model.name,
        task=model.task,
        target=dataset.target_name,
        n_samples=len(dataset),
        n_splits=N_SPLITS,
        seed=SEED,
        feature_names=dataset.feature_names,
        headline_metric=headline,
        per_fold=per_fold,
        mean=mean,
        std=std,
    )


def _subset(dataset: TabularDataset, idx: np.ndarray) -> TabularDataset:
    return TabularDataset(
        features=dataset.features.iloc[idx].reset_index(drop=True),
        target=dataset.target.iloc[idx].reset_index(drop=True),
        target_name=dataset.target_name,
    )


def split_dataset(dataset: TabularDataset, task: str) -> tuple[TabularDataset, TabularDataset]:
    """Seeded train/test split, stratified for classification."""
    indices = np.arange(len(dataset))
    stratify = dataset.target.to_numpy() if task == "classification" else None
    train_idx, test_idx = train_test_split(indices, test_size=TEST_SIZE, random_state=SEED, stratify=stratify)
    return _subset(dataset, train_idx), _subset(dataset, test_idx)


def evaluate_tabular(
    dataset: TabularDataset, model: TabularModel, *, holdout: bool
) -> tuple[EvaluationResult, BaseEstimator | None]:
    """Evaluate ``model`` on ``dataset``.

    Without ``holdout`` this is plain cross-validation on the whole dataset.
    With ``holdout`` the data is split first: cross-validation runs on the
    training split, a final model is fit on that split and scored once on the
    held-out test split, and that fitted model is returned for saving.
    """
    if not holdout:
        return cross_validate(dataset, model), None

    train_dataset, test_dataset = split_dataset(dataset, model.task)
    cv_result = cross_validate(train_dataset, model)

    estimator = model.build()
    estimator.fit(train_dataset.features.to_numpy(), train_dataset.target.to_numpy())
    prediction = _predict(estimator, test_dataset.features.to_numpy(), model.task)
    test_metrics = {k: float(v) for k, v in _score_fn(model.task)(test_dataset.target.to_numpy(), prediction).items()}

    result = replace(
        cv_result,
        test={
            "test_size": TEST_SIZE,
            "n_train": len(train_dataset),
            "n_test": len(test_dataset),
            "seed": SEED,
            "metrics": test_metrics,
        },
    )
    return result, estimator
