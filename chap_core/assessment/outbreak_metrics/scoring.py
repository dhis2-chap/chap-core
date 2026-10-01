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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from chap_core.assessment.outbreak_metrics import (
    BinaryConfusion,
    get_binary_outbreak_metrics,
    get_categorical_outbreak_metrics,
)
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


def score(labelled: pd.DataFrame, n_levels: int, group_by: list[str]) -> list[GroupScore]:
    """Run every registered outbreak metric over the labelled cells, per group.

    Args:
        labelled: Output of :func:`label_cells`.
        n_levels: Number of levels in the policy.
        group_by: Cell dimensions to group by, any of ``location``, ``time_period``
            and ``horizon_distance``. Empty pools every cell into one group.

    Returns:
        One :class:`GroupScore` per group, in sorted group order.
    """
    ranked = categories(labelled)
    if not group_by:
        return [_score_group({}, labelled, ranked, n_levels)]
    ranked_groups = dict(list(ranked.groupby(group_by)))
    return [
        _score_group(
            {
                dim: value.item() if isinstance(value, np.generic) else value
                for dim, value in zip(group_by, key, strict=True)
            },
            cells,
            ranked_groups[key],
            n_levels,
        )
        for key, cells in labelled.groupby(group_by, sort=True)
    ]


@dataclass
class GroupScore:
    """Outbreak metrics for one group of cells."""

    group: dict[str, Any]
    n_cells: int
    levels: list[dict[str, float | None]]
    categorical: dict[str, float | None]


def _score_group(group: dict[str, Any], cells: pd.DataFrame, ranked: pd.DataFrame, n_levels: int) -> GroupScore:
    binary_metrics = get_binary_outbreak_metrics()
    categorical_metrics = get_categorical_outbreak_metrics()
    levels = []
    for i in range(n_levels):
        confusion = binary_confusion(cells[cells["level"] == i])
        levels.append({metric_id: spec.compute(confusion) for metric_id, spec in binary_metrics.items()})
    matrix = confusion_matrix(ranked["observed"], ranked["predicted"], n_levels + 1)
    return GroupScore(
        group=group,
        n_cells=len(ranked),
        levels=levels,
        categorical={metric_id: spec.compute(matrix) for metric_id, spec in categorical_metrics.items()},
    )


def binary_confusion(cells: pd.DataFrame) -> BinaryConfusion:
    """Confusion counts of one level's labelled cells."""
    outbreak, alert = cells["outbreak"], cells["alert"]
    return BinaryConfusion(
        tp=int((outbreak & alert).sum()),
        fp=int((~outbreak & alert).sum()),
        fn=int((outbreak & ~alert).sum()),
        tn=int((~outbreak & ~alert).sum()),
    )


def confusion_matrix(observed: pd.Series, predicted: pd.Series, n_categories: int) -> np.ndarray:
    """Count of cells per (observed, predicted) category, observed along the rows."""
    matrix = np.zeros((n_categories, n_categories), dtype=int)
    np.add.at(matrix, (observed.to_numpy(dtype=int), predicted.to_numpy(dtype=int)), 1)
    return matrix
