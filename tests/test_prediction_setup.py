"""ORM-level tests for PredictionSetup relationships and cascade behavior.

Uses an in-memory SQLite engine with foreign-key enforcement enabled. The
alembic round-trip test verifies the production schema against PostgreSQL;
these tests verify the runtime ORM behavior independent of any specific DB.
"""

from __future__ import annotations

import datetime

import pytest
from sqlalchemy import event
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, SQLModel, create_engine, select

from chap_core.database.dataset_tables import DataSet
from chap_core.database.model_templates_and_config_tables import ConfiguredModelDB, ModelTemplateDB
from chap_core.database.tables import Backtest, Prediction, PredictionSetup


@pytest.fixture
def engine():
    eng = create_engine("sqlite://")

    @event.listens_for(eng, "connect")
    def _enable_fk(dbapi_connection, _connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    SQLModel.metadata.create_all(eng)
    return eng


def _make_parents(session: Session) -> tuple[int, int, int]:
    template = ModelTemplateDB(name="tpl")
    session.add(template)
    session.commit()
    assert template.id is not None

    model = ConfiguredModelDB(name="cfg", model_template_id=template.id)
    session.add(model)

    dataset = DataSet(name="ds")
    session.add(dataset)

    session.commit()
    assert model.id is not None
    assert dataset.id is not None

    backtest = Backtest(
        dataset_id=dataset.id,
        model_id="cfg",
        name="bt",
        model_db_id=model.id,
    )
    session.add(backtest)
    session.commit()
    assert backtest.id is not None
    return backtest.id, model.id, dataset.id


def _make_setup(backtest_id: int, configured_model_id: int, name: str = "setup") -> PredictionSetup:
    return PredictionSetup(
        name=name,
        created=datetime.datetime.now(),
        backtest_id=backtest_id,
        configured_model_id=configured_model_id,
        schedule_cron_expression="0 6 * * 1",
        schedule_enabled=True,
    )


def test_setup_links_to_backtest_in_both_directions(engine):
    with Session(engine) as session:
        backtest_id, model_id, _ = _make_parents(session)
        session.add(_make_setup(backtest_id, model_id))
        session.commit()

    with Session(engine) as session:
        backtest = session.exec(select(Backtest).where(Backtest.id == backtest_id)).one()
        assert backtest.prediction_setup is not None
        assert backtest.prediction_setup.name == "setup"

        setup = backtest.prediction_setup
        assert setup.backtest is not None
        assert setup.backtest.id == backtest_id


def test_schedule_fields_round_trip(engine):
    with Session(engine) as session:
        backtest_id, model_id, _ = _make_parents(session)
        setup = _make_setup(backtest_id, model_id)
        setup.schedule_cron_expression = "*/5 * * * *"
        setup.schedule_enabled = False
        session.add(setup)
        session.commit()
        session.refresh(setup)

        assert setup.schedule_cron_expression == "*/5 * * * *"
        assert setup.schedule_enabled is False


def test_backtest_can_have_at_most_one_setup(engine):
    with Session(engine) as session:
        backtest_id, model_id, _ = _make_parents(session)
        session.add(_make_setup(backtest_id, model_id, name="first"))
        session.commit()

        session.add(_make_setup(backtest_id, model_id, name="second"))
        with pytest.raises(IntegrityError):
            session.commit()


def test_deleting_backtest_cascades_to_setup(engine):
    with Session(engine) as session:
        backtest_id, model_id, _ = _make_parents(session)
        session.add(_make_setup(backtest_id, model_id))
        session.commit()

    with Session(engine) as session:
        backtest = session.exec(select(Backtest).where(Backtest.id == backtest_id)).one()
        session.delete(backtest)
        session.commit()

    with Session(engine) as session:
        assert session.exec(select(Backtest).where(Backtest.id == backtest_id)).first() is None
        assert session.exec(select(PredictionSetup).where(PredictionSetup.backtest_id == backtest_id)).first() is None


def test_deleting_setup_keeps_predictions_with_null_link(engine):
    with Session(engine) as session:
        backtest_id, model_id, dataset_id = _make_parents(session)
        setup = _make_setup(backtest_id, model_id)
        session.add(setup)
        session.commit()
        assert setup.id is not None
        setup_id = setup.id

        prediction = Prediction(
            dataset_id=dataset_id,
            model_id="cfg",
            n_periods=3,
            name="pred",
            created=datetime.datetime.now(),
            model_db_id=model_id,
            prediction_setup_id=setup_id,
        )
        session.add(prediction)
        session.commit()
        assert prediction.id is not None
        prediction_id = prediction.id

    with Session(engine) as session:
        setup = session.exec(select(PredictionSetup).where(PredictionSetup.id == setup_id)).one()
        session.delete(setup)
        session.commit()

    with Session(engine) as session:
        assert session.exec(select(PredictionSetup).where(PredictionSetup.id == setup_id)).first() is None
        surviving = session.exec(select(Prediction).where(Prediction.id == prediction_id)).one()
        assert surviving.prediction_setup_id is None


def test_deleting_backtest_keeps_predictions_with_null_setup_link(engine):
    """Transitive: backtest delete cascades to setup, but predictions survive
    with prediction_setup_id = NULL."""
    with Session(engine) as session:
        backtest_id, model_id, dataset_id = _make_parents(session)
        setup = _make_setup(backtest_id, model_id)
        session.add(setup)
        session.commit()
        assert setup.id is not None
        setup_id = setup.id

        prediction = Prediction(
            dataset_id=dataset_id,
            model_id="cfg",
            n_periods=3,
            name="pred",
            created=datetime.datetime.now(),
            model_db_id=model_id,
            prediction_setup_id=setup_id,
        )
        session.add(prediction)
        session.commit()
        assert prediction.id is not None
        prediction_id = prediction.id

    with Session(engine) as session:
        backtest = session.exec(select(Backtest).where(Backtest.id == backtest_id)).one()
        session.delete(backtest)
        session.commit()

    with Session(engine) as session:
        assert session.exec(select(Backtest).where(Backtest.id == backtest_id)).first() is None
        assert session.exec(select(PredictionSetup).where(PredictionSetup.id == setup_id)).first() is None
        surviving = session.exec(select(Prediction).where(Prediction.id == prediction_id)).one()
        assert surviving.prediction_setup_id is None
