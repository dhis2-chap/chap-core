"""Storing an evaluation's forecasts as rows, and reading a stored backtest back with column queries."""

import pandas as pd
from sqlalchemy import inspect
from sqlmodel import Session

from chap_core.assessment.evaluation import Evaluation
from chap_core.database.database import SessionWrapper
from chap_core.database.dataset_tables import DataSet, Observation
from chap_core.database.tables import Backtest, BacktestSpecification


def _sorted(df, keys):
    return pd.DataFrame(df).sort_values(keys).reset_index(drop=True)


def _store(engine, evaluation, observations):
    """Store the dataset, a specification and the evaluation's backtest; return the backtest id."""
    with Session(engine) as session:
        dataset = DataSet(name="stored", observations=observations)
        session.add(dataset)
        session.commit()
        specification = BacktestSpecification(dataset_id=dataset.id)
        session.add(specification)
        session.commit()
        backtest = Backtest(
            name="stored",
            dataset_id=dataset.id,
            specification_id=specification.id,
            model_id="model",
            model_db_id=1,
            org_units=evaluation.get_org_units(),
            split_periods=evaluation.get_split_periods(),
        )
        SessionWrapper(session=session).add_backtest(backtest, forecast_rows=evaluation.forecast_rows())
        return backtest.id


def test_stored_forecast_rows_read_back_through_from_backtest(file_engine, backtest, tmp_path):
    original = Evaluation.from_backtest(backtest)
    path = tmp_path / "evaluation.nc"
    original.to_file(path)
    evaluation = Evaluation.from_file(path)
    observations = [
        Observation(period=o.period, org_unit=o.org_unit, feature_name=o.feature_name, value=o.value)
        for o in backtest.dataset.observations
    ] + [Observation(period="2022-01", org_unit="OrgUnit1", feature_name="rainfall", value=99.0)]

    backtest_id = _store(file_engine, evaluation, observations)

    with Session(file_engine) as session:
        stored = session.get(Backtest, backtest_id)
        flat = Evaluation.from_backtest(stored).to_flat()
        # Read with column queries, not by loading the relationships.
        state = inspect(stored)
        assert state is not None
        assert {"forecasts", "dataset"} <= state.unloaded

    forecast_keys = ["location", "time_period", "horizon_distance", "sample"]
    expected = original.to_flat()
    pd.testing.assert_frame_equal(_sorted(flat.forecasts, forecast_keys), _sorted(expected.forecasts, forecast_keys))
    observation_keys = ["location", "time_period"]
    pd.testing.assert_frame_equal(
        _sorted(flat.observations, observation_keys), _sorted(expected.observations, observation_keys)
    )


def test_forecast_rows_match_the_backtest_forecasts(backtest, tmp_path):
    path = tmp_path / "evaluation.nc"
    Evaluation.from_backtest(backtest).to_file(path)

    rows = Evaluation.from_file(path).forecast_rows()
    forecasts = Evaluation.from_file(path).to_backtest().forecasts

    as_rows = [
        {
            "org_unit": f.org_unit,
            "period": f.period,
            "last_train_period": f.last_train_period,
            "last_seen_period": f.last_seen_period,
            "values": f.values,
        }
        for f in forecasts
    ]
    assert rows == as_rows
    assert len(rows) == len(backtest.forecasts)
