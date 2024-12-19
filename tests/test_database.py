import pytest
from sqlalchemy import create_engine
from sqlmodel import SQLModel, Session, select

from chap_core.database.tables import BackTest, DataSet
from chap_core.datatypes import HealthPopulationData
from chap_core.rest_api_src.db_worker_functions import run_backtest
from chap_core.testing.testing import assert_dataset_equal
from chap_core.database.database import SessionWrapper
import chap_core.database.database
from unittest.mock import patch

@pytest.fixture
def engine():
    engine = create_engine("sqlite://")
    SQLModel.metadata.create_all(engine)
    return engine


@pytest.fixture
def seeded_engine(engine, weekly_full_data):
    with SessionWrapper(engine) as session:
        session.add_dataset('full_data', weekly_full_data, 'polygons')
    return engine


def test_dataset_roundrip(health_population_data, engine):
    with SessionWrapper(engine) as session:
        dataset_id = session.add_dataset('health_population', health_population_data, 'polygons')
        dataset = session.get_dataset(dataset_id, HealthPopulationData)
        assert_dataset_equal(dataset, health_population_data)


def test_backtest(seeded_engine):
    with Session(seeded_engine) as session:
        dataset_id = session.exec(select(DataSet.id)).first()
    #with patch('chap_core.database.database.engine', seeded_engine):
    with SessionWrapper(seeded_engine) as session:
        res = run_backtest('naive_model', dataset_id, 12, 2, 1, session=session)
    # res = run_backtest('naive_model', dataset_id, 12, 2, 1)
    with Session(seeded_engine) as session:
        backtests = session.exec(select(BackTest)).all()
        assert len(backtests) == 1
        backtest = backtests[0]
        assert backtest.dataset_id == dataset_id
        assert len(backtest.forecasts) == 12 * 2 * 10
