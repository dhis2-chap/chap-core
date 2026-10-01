"""Score a backtest's forecasts against the levels of an alert policy.

Each level of an :class:`~chap_core.database.alert_tables.AlertPolicy` is judged as
its own binary classifier: a cell ``(location, time_period, horizon_distance)`` is an
outbreak at that level when the observed cases exceed the level's threshold, and is
alerted when the fraction of forecast samples above the threshold reaches the level's
``exceedance_threshold``.

The levels together also split every cell into ordered categories -- ``0`` for no
level, ``i + 1`` for the most severe level ``i`` breached -- so the policy can be judged
as a single ordinal classifier, ternary for a two-level policy.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from chap_core.assessment.thresholds import get_threshold_strategy

if TYPE_CHECKING:
    from chap_core.database.alert_tables import AlertLevel

_CELL = ["location", "time_period", "horizon_distance"]


def label_cells(
    historical_observations: pd.DataFrame,
    observations: pd.DataFrame,
    forecasts: pd.DataFrame,
    levels: list[AlertLevel],
) -> pd.DataFrame:
    """Label every scored cell as outbreak and alert at every level of a policy.

    Args:
        historical_observations: Observations the thresholds are computed from, with
            columns ``[location, time_period, disease_cases]``.
        observations: Observed cases to score against, same columns.
        forecasts: Forecast samples, with columns
            ``[location, time_period, horizon_distance, sample, forecast]``.
        levels: The policy's levels, least to most severe.

    Returns:
        One row per ``(location, time_period, horizon_distance, level)`` with boolean
        ``outbreak`` and ``alert`` columns. Cells without an observation, or without
        a threshold at every level, are dropped, so every level is scored on the
        same cells.

    Raises:
        ValueError: If a strategy cannot compute thresholds from the history.
    """
    period_ids = sorted(forecasts["time_period"].astype(str).unique())
    thresholds = pd.concat(
        [
            _level_thresholds(historical_observations, period_ids, level).assign(level=i)
            for i, level in enumerate(levels)
        ],
        ignore_index=True,
    )
    exceedance = pd.Series([level.exceedance_threshold for level in levels], name="exceedance_threshold")

    obs = observations[["location", "time_period", "disease_cases"]].dropna(subset=["disease_cases"])
    obs = obs.assign(time_period=obs["time_period"].astype(str))
    fc = forecasts.assign(time_period=forecasts["time_period"].astype(str))
    fc = fc.merge(thresholds, on=["location", "time_period"])
    fc["exceeds"] = fc["forecast"] > fc["threshold"]
    cells = fc.groupby([*_CELL, "level", "threshold"], as_index=False)["exceeds"].mean()
    cells = cells.merge(obs, on=["location", "time_period"])

    # groupby drops NaN thresholds, so a cell missing any level has fewer than len(levels) rows
    cells = cells[cells.groupby(_CELL)["level"].transform("nunique") == len(levels)]
    cells = cells.merge(exceedance, left_on="level", right_index=True)
    cells["outbreak"] = cells["disease_cases"] > cells["threshold"]
    cells["alert"] = cells["exceeds"] >= cells["exceedance_threshold"]
    return pd.DataFrame(cells[[*_CELL, "level", "outbreak", "alert"]]).reset_index(drop=True)


def _level_thresholds(historical_observations: pd.DataFrame, period_ids: list[str], level: AlertLevel) -> pd.DataFrame:
    """One level's threshold per ``(location, time_period)``, from its registered strategy."""
    strategy_cls = get_threshold_strategy(level.threshold_strategy)
    if strategy_cls is None:
        raise ValueError(f"Threshold strategy {level.threshold_strategy} is not registered")
    result = strategy_cls().compute(historical_observations, period_ids, level.threshold_params)
    return pd.DataFrame(result.rename(columns={"period_id": "time_period"})[["location", "time_period", "threshold"]])


def categories(labelled: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-level labels into one observed and one predicted category per cell.

    The category is ``0`` when no level is breached and ``i + 1`` for the most severe
    level ``i`` breached.
    """
    ranked = labelled.assign(
        observed=np.where(labelled["outbreak"], labelled["level"] + 1, 0),
        predicted=np.where(labelled["alert"], labelled["level"] + 1, 0),
    )
    return ranked.groupby(_CELL, as_index=False)[["observed", "predicted"]].max()


def binary_metrics(outbreak: pd.Series, alert: pd.Series) -> dict[str, float | None]:
    """Binary classification metrics of ``alert`` against ``outbreak``.

    A metric whose denominator is zero (for instance sensitivity when there are no
    outbreaks) is ``None``.
    """
    tp = int((outbreak & alert).sum())
    fp = int((~outbreak & alert).sum())
    fn = int((outbreak & ~alert).sum())
    tn = int((~outbreak & ~alert).sum())
    sensitivity = _ratio(tp, tp + fn)
    specificity = _ratio(tn, tn + fp)
    mcc_denominator = float(np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "accuracy": _ratio(tp + tn, tp + tn + fp + fn),
        "ppv": _ratio(tp, tp + fp),
        "npv": _ratio(tn, tn + fn),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "balanced_accuracy": None if sensitivity is None or specificity is None else (sensitivity + specificity) / 2,
        "mcc": _ratio(tp * tn - fp * fn, mcc_denominator),
    }


def categorical_metrics(observed: pd.Series, predicted: pd.Series, n_categories: int) -> dict[str, float | None]:
    """Multi-class metrics of the predicted category against the observed one.

    ``macro_f1`` averages F1 over the categories that occur in either series.
    ``weighted_kappa`` is Cohen's kappa with quadratic weights over all
    ``n_categories``, so a miss by one level costs less than a miss by two; it is
    ``None`` when undefined, e.g. when every cell falls in one category.
    """
    from sklearn.exceptions import UndefinedMetricWarning
    from sklearn.metrics import cohen_kappa_score, f1_score

    if observed.empty:
        return {"accuracy": None, "macro_f1": None, "weighted_kappa": None}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        kappa = cohen_kappa_score(observed, predicted, labels=list(range(n_categories)), weights="quadratic")
    return {
        "accuracy": float((observed == predicted).mean()),
        "macro_f1": float(f1_score(observed, predicted, average="macro", zero_division=0)),
        "weighted_kappa": None if np.isnan(kappa) else float(kappa),
    }


def _ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else numerator / denominator
