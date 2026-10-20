"""ORM-level tests for BacktestSpecification lifetime against its dataset.

Uses an in-memory SQLite engine with foreign-key enforcement enabled, the same
way tests/test_prediction_setup.py does, because the behaviour under test is a
database-level ON DELETE action rather than anything the ORM does in Python.
"""

from __future__ import annotations

import pytest
from sqlmodel import Session, select

from chap_core.database.dataset_tables import DataSet
from chap_core.database.tables import BacktestSpecification


def test_deleting_dataset_removes_specifications_that_outlived_their_backtests(engine):
    """A dataset whose backtests are gone is still deletable.

    Nothing deletes a specification on its own, and a backtest that fails after
    resolving one leaves it behind with no backtest at all, so without the cascade
    those leftover rows would keep the dataset pinned forever.
    """
    with Session(engine) as session:
        dataset = DataSet(name="ds")
        session.add(dataset)
        session.commit()
        assert dataset.id is not None
        dataset_id = dataset.id

        session.add(BacktestSpecification(dataset_id=dataset_id))
        session.commit()

        session.delete(dataset)
        session.commit()

        assert session.get(DataSet, dataset_id) is None
        remaining = session.exec(
            select(BacktestSpecification).where(BacktestSpecification.dataset_id == dataset_id)
        ).all()
        assert remaining == []
