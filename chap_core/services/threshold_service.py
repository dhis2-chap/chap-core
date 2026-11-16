"""Threshold lines computed from a stored dataset's history of a target column."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from chap_core.assessment.thresholds import get_threshold_strategy
from chap_core.database.dataset_manager import DataSetManager

if TYPE_CHECKING:
    from sqlmodel import Session

    from chap_core.assessment.thresholds.params import ThresholdParams


class UnknownStrategyError(Exception):
    """The params name a threshold strategy that is not registered."""


class NoObservationsError(Exception):
    """The dataset has no target observations for the requested locations."""


class InvalidThresholdInputError(Exception):
    """The strategy cannot use the supplied history or periods."""


def compute_thresholds(
    session: Session,
    dataset_id: int,
    period_ids: list[str],
    params: ThresholdParams,
    locations: list[str] | None = None,
    target_column: str = "disease_cases",
) -> pd.DataFrame:
    """Return period_id, location, line and threshold, including NaN for missing cells.

    Rows follow the requested period, location and line order. Omitted or empty
    locations select every location with target observations, in sorted order.
    Raises NoObservationsError for missing history, UnknownStrategyError for an unknown
    strategy and InvalidThresholdInputError for invalid history or periods.
    """
    strategy_cls = get_threshold_strategy(params.type)
    if strategy_cls is None:
        raise UnknownStrategyError(f"Threshold strategy {params.type} is in the request schema but not registered")

    rows = DataSetManager(session).feature_values(dataset_id, target_column, org_units=locations or None)
    if not rows:
        raise NoObservationsError(f"No {target_column} observations found for dataset {dataset_id}")
    # Strategies read the history from a disease_cases column whatever the target is.
    historical = pd.DataFrame(rows, columns=["time_period", "location", "disease_cases"])[
        ["location", "time_period", "disease_cases"]
    ]
    try:
        result = strategy_cls().compute(historical, period_ids, params)
    except ValueError as e:
        raise InvalidThresholdInputError(str(e)) from e
    locations = locations or sorted(historical["location"].unique())
    grid = pd.MultiIndex.from_product(
        [period_ids, locations, range(len(params.lines))], names=["period_id", "location", "line"]
    )
    return result.set_index(["period_id", "location", "line"])["threshold"].reindex(grid).reset_index()
