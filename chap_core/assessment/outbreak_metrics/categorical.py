"""Whole-policy outbreak metrics, computed from the confusion matrix of categories.

Rows of the matrix are observed categories and columns predicted ones; category
``0`` is no level and ``i + 1`` is level ``i``, so the order is the severity order.
"""

from __future__ import annotations

import numpy as np

from chap_core.assessment.outbreak_metrics import categorical_outbreak_metric
from chap_core.assessment.outbreak_metrics.binary import ratio


@categorical_outbreak_metric("accuracy", "Accuracy", "Share of cells whose predicted category is the observed one.")
def accuracy(confusion: np.ndarray) -> float | None:
    return ratio(float(np.trace(confusion)), float(confusion.sum()))


@categorical_outbreak_metric(
    "macro_f1", "Macro F1", "One-vs-rest F1 averaged over the categories that are observed or predicted."
)
def macro_f1(confusion: np.ndarray) -> float | None:
    totals = confusion.sum(axis=0) + confusion.sum(axis=1)
    present = totals > 0
    if not present.any():
        return None
    return float(np.mean(2 * np.diag(confusion)[present] / totals[present]))


@categorical_outbreak_metric(
    "weighted_kappa",
    "Weighted kappa",
    "Cohen's kappa with quadratic weights, so a miss by one level costs less than a miss by two. "
    "1 is perfect agreement, 0 is chance level.",
)
def weighted_kappa(confusion: np.ndarray) -> float | None:
    n_categories = confusion.shape[0]
    if n_categories < 2:
        return None
    index = np.arange(n_categories)
    weights = (index[:, None] - index[None, :]) ** 2 / (n_categories - 1) ** 2
    expected = np.outer(confusion.sum(axis=1), confusion.sum(axis=0)) / max(confusion.sum(), 1)
    disagreement = ratio(float((weights * confusion).sum()), float((weights * expected).sum()))
    return None if disagreement is None else 1 - disagreement
