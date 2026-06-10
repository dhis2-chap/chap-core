"""Equivalence tests for the database-backed facet dimensions.

These confirm that pushing coordinate filtering down to SQL (`load_filtered_flat_data`)
and enumerating coordinates with cheap queries (`distinct_values`) produce the same
result as the in-memory full-scan path the plots used before.
"""

from __future__ import annotations

import pandas as pd
import pytest
from sqlmodel import Session, SQLModel, create_engine

from chap_core.assessment.backtest_plots.db_dimensions import (
    HorizonDistanceDimension,
    LocationDimension,
    SplitPeriodDimension,
    load_filtered_flat_data,
)
from chap_core.assessment.backtest_plots.evaluation_plot import EvaluationPlot
from chap_core.assessment.backtest_plots.predicted_vs_actual_plot import PredictedVsActualPlot
from chap_core.assessment.evaluation import Evaluation

FORECAST_COLS = ["location", "time_period", "horizon_distance", "sample", "forecast"]


@pytest.fixture
def db_session(backtest):
    """Persist the in-memory `backtest` fixture into a throwaway SQLite database."""
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(backtest)
        session.commit()
        session.refresh(backtest)
        yield session


def _records(df: pd.DataFrame, cols: list[str]) -> set[tuple]:
    if df.empty:
        return set()
    return set(map(tuple, df[cols].itertuples(index=False, name=None)))


def test_location_coords_match_full_scan(db_session, backtest):
    full = Evaluation.from_backtest(backtest).to_flat()
    plotter = EvaluationPlot()
    full_coords = plotter.facet_coords(pd.DataFrame(full.observations), pd.DataFrame(full.forecasts))

    db_values = LocationDimension(field_name="location:N", display_name="Location").distinct_values(
        db_session, backtest
    )
    assert set(db_values) == set(full_coords["location"])


def test_split_period_coords_match_full_scan(db_session, backtest):
    full = Evaluation.from_backtest(backtest).to_flat()
    plotter = EvaluationPlot()
    full_coords = plotter.facet_coords(pd.DataFrame(full.observations), pd.DataFrame(full.forecasts))

    db_values = SplitPeriodDimension(field_name="split_period:O", display_name="Split Period").distinct_values(
        db_session, backtest
    )
    assert set(db_values) == set(full_coords["split_period"])


def test_horizon_coords_match_full_scan(db_session, backtest):
    full = Evaluation.from_backtest(backtest).to_flat()
    plotter = PredictedVsActualPlot()
    full_coords = plotter.facet_coords(pd.DataFrame(full.observations), pd.DataFrame(full.forecasts))

    db_values = HorizonDistanceDimension(
        field_name="horizon_distance:O", display_name="Horizon Distance"
    ).distinct_values(db_session, backtest)
    assert set(db_values) == set(full_coords["horizon_distance"])


def test_location_filter_matches_full_scan(db_session, backtest):
    full_forecasts = pd.DataFrame(Evaluation.from_backtest(backtest).to_flat().forecasts)
    location = sorted(full_forecasts["location"].unique())[0]
    dims = EvaluationPlot.facet_dimensions

    filtered = load_filtered_flat_data(db_session, backtest, {"location": location}, dims)

    expected = full_forecasts[full_forecasts["location"] == location]
    assert _records(pd.DataFrame(filtered.forecasts), FORECAST_COLS) == _records(expected, FORECAST_COLS)


def test_horizon_filter_matches_full_scan(db_session, backtest):
    full_forecasts = pd.DataFrame(Evaluation.from_backtest(backtest).to_flat().forecasts)
    horizon = int(sorted(full_forecasts["horizon_distance"].unique())[-1])
    dims = PredictedVsActualPlot.facet_dimensions

    filtered = load_filtered_flat_data(db_session, backtest, {"horizon_distance": horizon}, dims)

    expected = full_forecasts[full_forecasts["horizon_distance"] == horizon]
    assert _records(pd.DataFrame(filtered.forecasts), FORECAST_COLS) == _records(expected, FORECAST_COLS)


def test_split_periods_partition_forecasts(db_session, backtest):
    """The split-period filters partition the full forecast set: every row appears in
    exactly one split coordinate's load."""
    full_forecasts = pd.DataFrame(Evaluation.from_backtest(backtest).to_flat().forecasts)
    dims = EvaluationPlot.facet_dimensions
    split_dim = SplitPeriodDimension(field_name="split_period:O", display_name="Split Period")

    collected: list[set[tuple]] = []
    for split in split_dim.distinct_values(db_session, backtest):
        filtered = load_filtered_flat_data(db_session, backtest, {"split_period": split}, dims)
        collected.append(_records(pd.DataFrame(filtered.forecasts), FORECAST_COLS))

    for i in range(len(collected)):
        for j in range(i + 1, len(collected)):
            assert collected[i].isdisjoint(collected[j])

    union: set[tuple] = set().union(*collected) if collected else set()
    assert union == _records(full_forecasts, FORECAST_COLS)


def test_unknown_split_coord_returns_no_rows(db_session, backtest):
    dims = EvaluationPlot.facet_dimensions
    filtered = load_filtered_flat_data(db_session, backtest, {"split_period": "1900-01-01"}, dims)
    assert pd.DataFrame(filtered.forecasts).empty


def test_subplot_renders_from_db_filtered_data(db_session, backtest):
    import altair as alt

    full = Evaluation.from_backtest(backtest).to_flat()
    plotter = EvaluationPlot()
    coords = plotter.facet_coords(pd.DataFrame(full.observations), pd.DataFrame(full.forecasts))
    selection = {"split_period": coords["split_period"][0], "location": coords["location"][0]}

    filtered = load_filtered_flat_data(db_session, backtest, selection, EvaluationPlot.facet_dimensions)
    chart = plotter.get_subplot(pd.DataFrame(filtered.observations), pd.DataFrame(filtered.forecasts), selection)
    assert isinstance(chart, alt.TopLevelMixin)
    assert not isinstance(chart, alt.FacetChart)
