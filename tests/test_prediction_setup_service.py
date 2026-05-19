"""Unit tests for chap_core.services.prediction_setup_service.

Uses an in-memory SQLite engine with foreign-key enforcement enabled so the
DB-level cascade and unique-constraint semantics are real, not mocked. These
tests cover the service in isolation; the HTTP-layer wiring is verified
separately by the FastAPI integration tests.
"""

from __future__ import annotations

import pytest
from sqlalchemy import event
from sqlmodel import Session, SQLModel, create_engine, select

from chap_core.database.dataset_tables import DataSet
from chap_core.database.model_templates_and_config_tables import ConfiguredModelDB, ModelTemplateDB
from chap_core.database.tables import Backtest, Prediction, PredictionSetup, QuantileTarget
from chap_core.services.prediction_setup_service import (
    BacktestNotFoundError,
    DuplicateSetupError,
    InvalidSetupError,
    PredictionSetupNotFoundError,
    create_prediction_setup,
    delete_prediction_setup,
    get_prediction_setup,
    list_prediction_setups,
    update_prediction_setup,
)


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

    backtest = Backtest(dataset_id=dataset.id, model_id="cfg", name="bt", model_db_id=model.id)
    session.add(backtest)
    session.commit()
    assert backtest.id is not None
    return backtest.id, model.id, dataset.id


def _create_default_setup(session: Session, backtest_id: int) -> PredictionSetup:
    return create_prediction_setup(
        session,
        backtest_id=backtest_id,
        name="setup",
        schedule_cron_expression="0 6 * * 1",
        schedule_enabled=True,
        quantile_targets=[QuantileTarget(quantile="median", data_element_id="DE_MED")],
    )


def test_create_happy_path_snapshots_backtest_dataset(engine):
    with Session(engine) as session:
        backtest_id, model_id, _ = _make_parents(session)
        setup = _create_default_setup(session, backtest_id)

        assert setup.id is not None
        assert setup.backtest_id == backtest_id
        assert setup.configured_model_id == model_id
        assert setup.schedule_cron_expression == "0 6 * * 1"
        assert setup.schedule_enabled is True
        assert len(setup.quantile_targets) == 1
        assert setup.quantile_targets[0].quantile == "median"


def test_create_with_missing_backtest_raises_not_found(engine):
    with Session(engine) as session:
        with pytest.raises(BacktestNotFoundError):
            create_prediction_setup(
                session,
                backtest_id=99999,
                name="ghost",
                schedule_cron_expression=None,
                schedule_enabled=False,
                quantile_targets=[],
            )


def test_create_with_empty_name_raises_invalid(engine):
    with Session(engine) as session:
        backtest_id, _, _ = _make_parents(session)
        with pytest.raises(InvalidSetupError, match="name"):
            create_prediction_setup(
                session,
                backtest_id=backtest_id,
                name="",
                schedule_cron_expression=None,
                schedule_enabled=False,
                quantile_targets=[],
            )


def test_create_with_invalid_cron_raises_invalid(engine):
    with Session(engine) as session:
        backtest_id, _, _ = _make_parents(session)
        with pytest.raises(InvalidSetupError, match="cron"):
            create_prediction_setup(
                session,
                backtest_id=backtest_id,
                name="setup",
                schedule_cron_expression="not a cron expression",
                schedule_enabled=False,
                quantile_targets=[],
            )


def test_create_with_enabled_but_no_expression_raises_invalid(engine):
    with Session(engine) as session:
        backtest_id, _, _ = _make_parents(session)
        with pytest.raises(InvalidSetupError, match="Enabled schedules require"):
            create_prediction_setup(
                session,
                backtest_id=backtest_id,
                name="setup",
                schedule_cron_expression=None,
                schedule_enabled=True,
                quantile_targets=[],
            )


def test_create_with_no_schedule_is_allowed(engine):
    with Session(engine) as session:
        backtest_id, _, _ = _make_parents(session)
        setup = create_prediction_setup(
            session,
            backtest_id=backtest_id,
            name="no-schedule",
            schedule_cron_expression=None,
            schedule_enabled=False,
            quantile_targets=[],
        )
        assert setup.schedule_cron_expression is None
        assert setup.schedule_enabled is False


def test_create_twice_on_same_backtest_raises_duplicate(engine):
    with Session(engine) as session:
        backtest_id, _, _ = _make_parents(session)
        _create_default_setup(session, backtest_id)
        with pytest.raises(DuplicateSetupError):
            _create_default_setup(session, backtest_id)


def test_get_returns_setup(engine):
    with Session(engine) as session:
        backtest_id, _, _ = _make_parents(session)
        created = _create_default_setup(session, backtest_id)

    with Session(engine) as session:
        assert created.id is not None
        fetched = get_prediction_setup(session, created.id)
        assert fetched.id == created.id
        assert fetched.name == "setup"


def test_get_with_missing_id_raises_not_found(engine):
    with Session(engine) as session:
        with pytest.raises(PredictionSetupNotFoundError):
            get_prediction_setup(session, 99999)


def test_list_returns_all_setups(engine):
    with Session(engine) as session:
        backtest_id, _, _ = _make_parents(session)
        _create_default_setup(session, backtest_id)

    with Session(engine) as session:
        setups = list_prediction_setups(session)
        assert len(setups) == 1
        assert setups[0].backtest_id == backtest_id


def test_update_name_only(engine):
    with Session(engine) as session:
        backtest_id, _, _ = _make_parents(session)
        setup = _create_default_setup(session, backtest_id)
        assert setup.id is not None
        updated = update_prediction_setup(session, setup.id, {"name": "renamed"})
        assert updated.name == "renamed"
        assert updated.schedule_cron_expression == "0 6 * * 1"


def test_update_schedule_clears_expression_when_disabling(engine):
    with Session(engine) as session:
        backtest_id, _, _ = _make_parents(session)
        setup = _create_default_setup(session, backtest_id)
        assert setup.id is not None
        updated = update_prediction_setup(
            session,
            setup.id,
            {"schedule_cron_expression": None, "schedule_enabled": False},
        )
        assert updated.schedule_cron_expression is None
        assert updated.schedule_enabled is False


def test_update_quantile_targets_replaces_list(engine):
    with Session(engine) as session:
        backtest_id, _, _ = _make_parents(session)
        setup = _create_default_setup(session, backtest_id)
        assert setup.id is not None
        updated = update_prediction_setup(
            session,
            setup.id,
            {"quantile_targets": [QuantileTarget(quantile="p90", data_element_id="DE_P90")]},
        )
        assert len(updated.quantile_targets) == 1
        assert updated.quantile_targets[0].quantile == "p90"


def test_update_with_immutable_field_raises_invalid(engine):
    with Session(engine) as session:
        backtest_id, _, _ = _make_parents(session)
        setup = _create_default_setup(session, backtest_id)
        assert setup.id is not None
        with pytest.raises(InvalidSetupError, match="immutable"):
            update_prediction_setup(session, setup.id, {"backtest_id": 999})


def test_update_with_null_name_raises_invalid(engine):
    with Session(engine) as session:
        backtest_id, _, _ = _make_parents(session)
        setup = _create_default_setup(session, backtest_id)
        assert setup.id is not None
        with pytest.raises(InvalidSetupError, match="name"):
            update_prediction_setup(session, setup.id, {"name": None})


def test_update_enabling_without_expression_raises_invalid(engine):
    with Session(engine) as session:
        backtest_id, _, _ = _make_parents(session)
        setup = create_prediction_setup(
            session,
            backtest_id=backtest_id,
            name="setup",
            schedule_cron_expression=None,
            schedule_enabled=False,
            quantile_targets=[],
        )
        assert setup.id is not None
        with pytest.raises(InvalidSetupError, match="Enabled schedules require"):
            update_prediction_setup(session, setup.id, {"schedule_enabled": True})


def test_update_with_missing_id_raises_not_found(engine):
    with Session(engine) as session:
        with pytest.raises(PredictionSetupNotFoundError):
            update_prediction_setup(session, 99999, {"name": "ghost"})


def test_delete_removes_setup_and_keeps_predictions(engine):
    with Session(engine) as session:
        backtest_id, model_id, dataset_id = _make_parents(session)
        setup = _create_default_setup(session, backtest_id)
        assert setup.id is not None
        setup_id = setup.id

        import datetime as _dt

        prediction = Prediction(
            dataset_id=dataset_id,
            model_id="cfg",
            n_periods=3,
            name="pred",
            created=_dt.datetime.now(),
            model_db_id=model_id,
            prediction_setup_id=setup_id,
        )
        session.add(prediction)
        session.commit()
        assert prediction.id is not None
        prediction_id = prediction.id

    with Session(engine) as session:
        delete_prediction_setup(session, setup_id)

    with Session(engine) as session:
        assert session.exec(select(PredictionSetup).where(PredictionSetup.id == setup_id)).first() is None
        surviving = session.exec(select(Prediction).where(Prediction.id == prediction_id)).one()
        assert surviving.prediction_setup_id is None


def test_delete_with_missing_id_raises_not_found(engine):
    with Session(engine) as session:
        with pytest.raises(PredictionSetupNotFoundError):
            delete_prediction_setup(session, 99999)


def test_session_wrapper_add_predictions_links_to_setup(engine):
    """SessionWrapper.add_predictions wires prediction_setup_id through to the Prediction row.

    Worker callers (run_prediction, predict_pipeline_from_composite_dataset) pass this id
    through; the test verifies the plumbing without exercising the full forecasting pipeline.
    """
    from types import SimpleNamespace

    import numpy as np

    from chap_core.database.database import SessionWrapper

    with Session(engine) as session:
        backtest_id, _, dataset_id = _make_parents(session)
        setup = _create_default_setup(session, backtest_id)
        assert setup.id is not None
        setup_id = setup.id

    class _Forecast:
        def __init__(self, periods, samples):
            self.time_period = periods
            self.samples = samples

        def __len__(self):
            return len(self.time_period)

    fake_period = SimpleNamespace(id="2024-01")
    predictions = {"loc_1": _Forecast([fake_period], [np.array([1.0, 2.0, 3.0])])}

    with SessionWrapper(engine) as wrapper:
        prediction_id = wrapper.add_predictions(
            predictions,
            dataset_id,
            "cfg",
            "wired",
            prediction_setup_id=setup_id,
        )

    with Session(engine) as session:
        prediction = session.exec(select(Prediction).where(Prediction.id == prediction_id)).one()
        assert prediction.prediction_setup_id == setup_id
