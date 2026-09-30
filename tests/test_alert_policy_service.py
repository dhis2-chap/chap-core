"""Unit tests for chap_core.services.alert_policy_service.

Uses an in-memory SQLite engine with foreign-key enforcement enabled so the
in-use guard on delete is real, not mocked. These tests cover the service in
isolation; the HTTP-layer wiring is verified by the FastAPI integration tests.
"""

from __future__ import annotations

import pytest
from sqlalchemy import event
from sqlmodel import Session, SQLModel, create_engine

from chap_core.assessment.thresholds.params import PercentileParams, SeasonalParams
from chap_core.database.alert_tables import AlertLevel
from chap_core.database.dataset_tables import DataSet
from chap_core.database.model_templates_and_config_tables import ConfiguredModelDB, ModelTemplateDB
from chap_core.database.tables import Backtest, BacktestSpecification
from chap_core.services.alert_policy_service import (
    AlertPolicyInUseError,
    AlertPolicyNotFoundError,
    InvalidAlertPolicyError,
    create_alert_policy,
    delete_alert_policy,
    get_alert_policy,
    list_alert_policies,
)
from chap_core.services.prediction_setup_service import create_prediction_setup


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


@pytest.fixture
def monitor_level():
    """A low tier drawn with the seasonal strategy."""
    return AlertLevel(
        name="monitor",
        threshold_params=SeasonalParams(type="seasonal", std_multiplier=1.0),
        exceedance_threshold=0.3,
    )


@pytest.fixture
def action_level():
    """A high tier drawn with the WHO percentile strategy."""
    return AlertLevel(
        name="action",
        threshold_params=PercentileParams(type="percentile", quantile=0.9),
        exceedance_threshold=0.8,
    )


def _make_backtest(session: Session) -> int:
    template = ModelTemplateDB(name="tpl", version="1.0.0")
    session.add(template)
    session.commit()
    assert template.id is not None

    model = ConfiguredModelDB(name="cfg", model_template_id=template.id)
    session.add(model)
    dataset = DataSet(name="ds")
    session.add(dataset)
    session.commit()
    assert model.id is not None and dataset.id is not None

    specification = BacktestSpecification(dataset_id=dataset.id)
    session.add(specification)
    session.commit()
    backtest = Backtest(
        dataset_id=dataset.id, model_id="cfg", name="bt", model_db_id=model.id, specification=specification
    )
    session.add(backtest)
    session.commit()
    assert backtest.id is not None
    return backtest.id


def test_create_round_trips_the_levels(engine, monitor_level, action_level):
    """Levels survive the JSON column, discriminated params and all."""
    with Session(engine) as session:
        policy = create_alert_policy(session, name="Dengue ladder", levels=[monitor_level, action_level])
        assert policy.id is not None

        stored = get_alert_policy(session, policy.id)
        assert [level.name for level in stored.levels] == ["monitor", "action"]
        assert stored.levels[1].exceedance_threshold == 0.8
        # The discriminator picked the right union member back out of the JSON column.
        params = stored.levels[1].threshold_params
        assert isinstance(params, PercentileParams)
        assert params.quantile == 0.9


def test_a_level_reports_the_strategy_from_its_params(monitor_level, action_level):
    """The strategy is read off the params discriminator, never stored twice."""
    assert monitor_level.threshold_strategy == "seasonal"
    assert action_level.threshold_strategy == "percentile"


def test_create_without_levels_is_rejected(engine):
    """A policy with no tiers can never raise anything."""
    with Session(engine) as session:
        with pytest.raises(InvalidAlertPolicyError, match="at least one level"):
            create_alert_policy(session, name="empty", levels=[])


def test_create_with_duplicate_level_names_is_rejected(engine, monitor_level):
    with Session(engine) as session:
        with pytest.raises(InvalidAlertPolicyError, match="unique"):
            create_alert_policy(session, name="dupes", levels=[monitor_level, monitor_level])


def test_a_level_with_several_threshold_lines_is_rejected(engine, action_level):
    """A level is judged against one line, so a band like `[0.25, 0.75]` is ambiguous."""
    band = action_level.model_dump() | {"threshold_params": {"type": "percentile", "quantile": [0.25, 0.75]}}
    with Session(engine) as session:
        with pytest.raises(InvalidAlertPolicyError, match="one threshold line"):
            create_alert_policy(session, name="band", levels=[band])


def test_create_with_empty_name_is_rejected(engine, monitor_level):
    with Session(engine) as session:
        with pytest.raises(InvalidAlertPolicyError, match="name is required"):
            create_alert_policy(session, name="  ", levels=[monitor_level])


def test_get_missing_policy_raises_not_found(engine):
    with Session(engine) as session:
        with pytest.raises(AlertPolicyNotFoundError):
            get_alert_policy(session, 99999)


def test_list_returns_every_policy(engine, monitor_level):
    with Session(engine) as session:
        create_alert_policy(session, name="first", levels=[monitor_level])
        create_alert_policy(session, name="second", levels=[monitor_level])
        assert sorted(policy.name for policy in list_alert_policies(session)) == ["first", "second"]


def test_delete_removes_an_unused_policy(engine, monitor_level):
    with Session(engine) as session:
        policy = create_alert_policy(session, name="p", levels=[monitor_level])
        assert policy.id is not None

        delete_alert_policy(session, policy.id)
        assert list_alert_policies(session) == []


def test_delete_is_refused_while_a_setup_points_at_it(engine, monitor_level):
    """A running forecast must not lose the definition of what counts as an alert."""
    with Session(engine) as session:
        policy = create_alert_policy(session, name="p", levels=[monitor_level])
        assert policy.id is not None
        backtest_id = _make_backtest(session)
        create_prediction_setup(
            session,
            backtest_id=backtest_id,
            name="setup",
            schedule_cron_expression=None,
            schedule_enabled=False,
            quantile_targets=[],
            alert_policy_id=policy.id,
        )

        with pytest.raises(AlertPolicyInUseError):
            delete_alert_policy(session, policy.id)


def test_delete_is_refused_while_alerts_reference_it(engine, monitor_level):
    """An alert's level name only means something against its policy."""
    from chap_core.database.alert_tables import Alert
    from chap_core.services.alert_service import create_alerts

    with Session(engine) as session:
        policy = create_alert_policy(session, name="p", levels=[monitor_level])
        assert policy.id is not None
        create_alerts(
            session,
            [Alert(time_period="2024-07", org_unit="A", alert_policy_id=policy.id, level="monitor")],
        )

        with pytest.raises(AlertPolicyInUseError, match="alerts raised against it"):
            delete_alert_policy(session, policy.id)
