"""Database-backed facet dimensions for backtest plots.

These dimensions know how to (a) enumerate their available coordinate values with
cheap database queries and (b) translate a requested coordinate into SQL filters,
so a single subplot can be served by loading only the matching forecast rows
instead of flattening the entire backtest in memory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import false, tuple_
from sqlmodel import select

from chap_core.assessment.backtest_plots import FacetDimension
from chap_core.assessment.flat_representations import (
    FlatForecasts,
    FlatObserved,
    convert_backtest_observations_to_flat_observations,
    convert_backtest_to_flat_forecasts,
    horizon_diff,
)
from chap_core.database.dataset_tables import Observation
from chap_core.database.tables import BacktestForecast
from chap_core.plotting.backtest_plot import clean_time
from chap_core.time_period import TimePeriod

if TYPE_CHECKING:
    from sqlalchemy.sql.elements import ColumnElement
    from sqlmodel import Session

    from chap_core.assessment.evaluation import FlatEvaluationData
    from chap_core.database.tables import Backtest


class DBFacetDimension(FacetDimension):
    """A facet dimension whose coordinates and filters can be served from the database."""

    def distinct_values(self, session: Session, backtest: Backtest) -> list[Any]:
        """Coordinate values available for this dimension, in display form."""
        raise NotImplementedError

    def forecast_filters(self, value: Any, session: Session, backtest: Backtest) -> list[ColumnElement]:
        """SQL clauses narrowing `BacktestForecast` to the rows for `value`."""
        return []

    def observation_filters(self, value: Any, session: Session, backtest: Backtest) -> list[ColumnElement]:
        """SQL clauses narrowing `Observation` to the rows for `value`."""
        return []


class LocationDimension(DBFacetDimension):
    """Facet by org unit (`location`), matching `BacktestForecast.org_unit` directly."""

    def distinct_values(self, session: Session, backtest: Backtest) -> list[Any]:
        if backtest.org_units:
            return sorted(backtest.org_units)
        rows = session.exec(
            select(BacktestForecast.org_unit).where(BacktestForecast.backtest_id == backtest.id).distinct()
        ).all()
        return sorted(rows)

    def forecast_filters(self, value: Any, session: Session, backtest: Backtest) -> list[ColumnElement]:
        return [BacktestForecast.org_unit == value]

    def observation_filters(self, value: Any, session: Session, backtest: Backtest) -> list[ColumnElement]:
        return [Observation.org_unit == value]


def _split_display(last_seen_period: str) -> str:
    """The split-period coordinate a forecast with this `last_seen_period` falls under.

    Mirrors the plot's `clean_time(time_period - horizon * delta)`, which reduces to
    `clean_time(last_seen_period - delta)`.
    """
    tp = TimePeriod.parse(last_seen_period)
    return str(clean_time((tp - tp.time_delta).to_string()))


def _distinct_last_seen(session: Session, backtest: Backtest) -> list[str]:
    return list(
        session.exec(
            select(BacktestForecast.last_seen_period).where(BacktestForecast.backtest_id == backtest.id).distinct()
        ).all()
    )


class SplitPeriodDimension(DBFacetDimension):
    """Facet by split period, derived from `BacktestForecast.last_seen_period`."""

    def distinct_values(self, session: Session, backtest: Backtest) -> list[Any]:
        return sorted({_split_display(ls) for ls in _distinct_last_seen(session, backtest)})

    def forecast_filters(self, value: Any, session: Session, backtest: Backtest) -> list[ColumnElement]:
        matching = [ls for ls in _distinct_last_seen(session, backtest) if _split_display(ls) == value]
        if not matching:
            return [false()]
        return [BacktestForecast.last_seen_period.in_(matching)]  # type: ignore[attr-defined]


def _distinct_period_pairs(session: Session, backtest: Backtest) -> list[tuple[str, str]]:
    rows = session.exec(
        select(BacktestForecast.period, BacktestForecast.last_seen_period)
        .where(BacktestForecast.backtest_id == backtest.id)
        .distinct()
    ).all()
    return [(str(period), str(last_seen)) for period, last_seen in rows]


class HorizonDistanceDimension(DBFacetDimension):
    """Facet by 1-based horizon distance.

    Coordinates run `1..max_horizon_distance` when the backtest records it, falling
    back to deriving distinct horizons from the forecasts for legacy rows.
    """

    def distinct_values(self, session: Session, backtest: Backtest) -> list[Any]:
        if backtest.max_horizon_distance is not None:
            return list(range(1, backtest.max_horizon_distance + 1))
        horizons = {horizon_diff(period, last_seen) for period, last_seen in _distinct_period_pairs(session, backtest)}
        return sorted(horizons)

    def forecast_filters(self, value: Any, session: Session, backtest: Backtest) -> list[ColumnElement]:
        pairs = [
            (period, last_seen)
            for period, last_seen in _distinct_period_pairs(session, backtest)
            if horizon_diff(period, last_seen) == int(value)
        ]
        if not pairs:
            return [false()]
        return [tuple_(BacktestForecast.period, BacktestForecast.last_seen_period).in_(pairs)]  # type: ignore[arg-type]


def load_filtered_flat_data(
    session: Session,
    backtest: Backtest,
    coords: dict[str, Any],
    dimensions: list[FacetDimension],
) -> FlatEvaluationData:
    """Load only the forecast/observation rows matching `coords`, as flat frames.

    Coordinates whose dimension is a `DBFacetDimension` are pushed down to SQL; any
    other coordinate is ignored here and left to the plot's in-memory filter.
    """
    from chap_core.assessment.evaluation import FlatEvaluationData

    dim_by_name = {d.clean_name: d for d in dimensions if isinstance(d, DBFacetDimension)}

    forecast_clauses: list[Any] = [BacktestForecast.backtest_id == backtest.id]
    observation_clauses: list[Any] = [
        Observation.dataset_id == backtest.dataset_id,
        Observation.feature_name == "disease_cases",
    ]
    for name, value in coords.items():
        dim = dim_by_name.get(name)
        if dim is None:
            continue
        forecast_clauses += dim.forecast_filters(value, session, backtest)
        observation_clauses += dim.observation_filters(value, session, backtest)

    forecasts = session.exec(select(BacktestForecast).where(*forecast_clauses)).all()
    observations = session.exec(select(Observation).where(*observation_clauses)).all()

    forecasts_df = convert_backtest_to_flat_forecasts(list(forecasts))
    observations_df = convert_backtest_observations_to_flat_observations(list(observations))

    return FlatEvaluationData(
        forecasts=FlatForecasts(forecasts_df),
        observations=FlatObserved(observations_df),
    )
