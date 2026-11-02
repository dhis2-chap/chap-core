"""Per-level outbreak metrics, computed from a level's :class:`BinaryConfusion`."""

from __future__ import annotations

import math

from chap_core.assessment.outbreak_metrics import BinaryConfusion, binary_outbreak_metric


def ratio(numerator: float, denominator: float) -> float | None:
    """``numerator / denominator``, or ``None`` when the denominator is zero."""
    return None if denominator == 0 else numerator / denominator


@binary_outbreak_metric("sensitivity", "Sensitivity", "Share of outbreaks that were alerted: TP / (TP + FN).")
def sensitivity(c: BinaryConfusion) -> float | None:
    return ratio(c.tp, c.tp + c.fn)


@binary_outbreak_metric("specificity", "Specificity", "Share of non-outbreaks that were not alerted: TN / (TN + FP).")
def specificity(c: BinaryConfusion) -> float | None:
    return ratio(c.tn, c.tn + c.fp)


@binary_outbreak_metric("accuracy", "Accuracy", "Share of cells where alert and outbreak agree.")
def accuracy(c: BinaryConfusion) -> float | None:
    return ratio(c.tp + c.tn, c.tp + c.fp + c.fn + c.tn)


@binary_outbreak_metric("ppv", "PPV", "Positive predictive value: share of alerts that were outbreaks, TP / (TP + FP).")
def ppv(c: BinaryConfusion) -> float | None:
    return ratio(c.tp, c.tp + c.fp)


@binary_outbreak_metric(
    "npv", "NPV", "Negative predictive value: share of non-alerts that were not outbreaks, TN / (TN + FN)."
)
def npv(c: BinaryConfusion) -> float | None:
    return ratio(c.tn, c.tn + c.fn)


@binary_outbreak_metric("f1", "F1", "Harmonic mean of sensitivity and PPV: 2TP / (2TP + FP + FN).")
def f1(c: BinaryConfusion) -> float | None:
    return ratio(2 * c.tp, 2 * c.tp + c.fp + c.fn)


@binary_outbreak_metric("balanced_accuracy", "Balanced accuracy", "Mean of sensitivity and specificity.")
def balanced_accuracy(c: BinaryConfusion) -> float | None:
    sens, spec = sensitivity(c), specificity(c)
    return None if sens is None or spec is None else (sens + spec) / 2


@binary_outbreak_metric(
    "mcc", "MCC", "Matthews correlation coefficient between alert and outbreak, in [-1, 1]; 0 is chance level."
)
def mcc(c: BinaryConfusion) -> float | None:
    return ratio(
        c.tp * c.tn - c.fp * c.fn, math.sqrt(float(c.tp + c.fp) * (c.tp + c.fn) * (c.tn + c.fp) * (c.tn + c.fn))
    )
