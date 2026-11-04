"""Unit tests for chap_core.services.threshold_policy_service.

Uses an in-memory SQLite engine with foreign-key enforcement enabled so the
in-use guard on delete is real, not mocked. These tests cover the service in
isolation; the HTTP-layer wiring is verified by the FastAPI integration tests.
"""

from __future__ import annotations

import pytest
from sqlmodel import Session

from chap_core.assessment.thresholds.params import PercentileParams, SeasonalParams
from chap_core.database.signal_tables import ThresholdLevel
from chap_core.database.dataset_tables import DataSet
from chap_core.database.model_templates_and_config_tables import ConfiguredModelDB, ModelTemplateDB
from chap_core.database.tables import Backtest, BacktestSpecification
from chap_core.services.threshold_policy_service import (
    ThresholdPolicyInUseError,
    ThresholdPolicyNotFoundError,
    InvalidThresholdPolicyError,
    create_threshold_policy,
    delete_threshold_policy,
    get_threshold_policy,
    list_threshold_policies,
)
from chap_core.services.prediction_setup_service import create_prediction_setup


@pytest.fixture
def level_1_level():
    """A low tier drawn with the seasonal strategy."""
    return ThresholdLevel(
        name="level_1",
        output_code="1",
        threshold_params=SeasonalParams(type="seasonal", std_multiplier=1.0),
        alarm_threshold=0.3,
    )


@pytest.fixture
def level_2_level():
    """A high tier drawn with the WHO percentile strategy."""
    return ThresholdLevel(
        name="level_2",
        output_code="2",
        threshold_params=PercentileParams(type="percentile", quantile=0.9),
        alarm_threshold=0.8,
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


def test_create_round_trips_the_levels(engine, level_1_level, level_2_level):
    """Levels survive the JSON column, discriminated params and all."""
    with Session(engine) as session:
        policy = create_threshold_policy(
            session, name="Dengue ladder", levels=[level_1_level, level_2_level], no_signal_output_code="GREEN"
        )
        assert policy.id is not None

        stored = get_threshold_policy(session, policy.id)
        assert [level.name for level in stored.levels] == ["level_1", "level_2"]
        assert stored.no_signal_output_code == "GREEN"
        assert [level.output_code for level in stored.levels] == ["1", "2"]
        assert stored.levels[1].alarm_threshold == 0.8
        # The discriminator picked the right union member back out of the JSON column.
        params = stored.levels[1].threshold_params
        assert isinstance(params, PercentileParams)
        assert params.quantile == 0.9


def test_a_level_reports_the_strategy_from_its_params(level_1_level, level_2_level):
    """The strategy is read off the params discriminator, never stored twice."""
    assert level_1_level.threshold_strategy == "seasonal"
    assert level_2_level.threshold_strategy == "percentile"


def test_create_defaults_omitted_codes_and_preserves_explicit_codes(engine, level_1_level, level_2_level):
    with Session(engine) as session:
        policy = create_threshold_policy(
            session,
            name="Mixed codes",
            levels=[
                level_1_level.model_dump(exclude={"output_code"}),
                level_2_level.model_copy(update={"output_code": "RED"}),
            ],
        )
        assert policy.no_signal_output_code == "0"
        assert [level.output_code for level in policy.levels] == ["1", "RED"]


def test_create_without_levels_is_rejected(engine):
    """A policy with no tiers can never raise anything."""
    with Session(engine) as session:
        with pytest.raises(InvalidThresholdPolicyError, match="at least one level"):
            create_threshold_policy(session, name="empty", levels=[])


def test_create_with_duplicate_level_names_is_rejected(engine, level_1_level):
    with Session(engine) as session:
        with pytest.raises(InvalidThresholdPolicyError, match="unique"):
            create_threshold_policy(session, name="dupes", levels=[level_1_level, level_1_level])


def test_a_level_name_with_surrounding_whitespace_is_rejected(engine, level_1_level):
    """Signals match level names exactly, so `"signal "` would never match `"signal"`."""
    padded = level_1_level.model_copy(update={"name": "level_1 "})
    with Session(engine) as session:
        with pytest.raises(InvalidThresholdPolicyError, match="whitespace"):
            create_threshold_policy(session, name="padded", levels=[padded])


def test_a_level_with_several_threshold_lines_is_rejected(engine, level_2_level):
    """A level is judged against one line, so a band like `[0.25, 0.75]` is ambiguous."""
    band = level_2_level.model_dump() | {"threshold_params": {"type": "percentile", "quantile": [0.25, 0.75]}}
    with Session(engine) as session:
        with pytest.raises(InvalidThresholdPolicyError, match="one threshold line"):
            create_threshold_policy(session, name="band", levels=[band])


def test_create_with_empty_name_is_rejected(engine, level_1_level):
    with Session(engine) as session:
        with pytest.raises(InvalidThresholdPolicyError, match="name is required"):
            create_threshold_policy(session, name="  ", levels=[level_1_level])


def test_get_missing_policy_raises_not_found(engine):
    with Session(engine) as session:
        with pytest.raises(ThresholdPolicyNotFoundError):
            get_threshold_policy(session, 99999)


def test_list_returns_every_policy(engine, level_1_level):
    with Session(engine) as session:
        create_threshold_policy(session, name="first", levels=[level_1_level])
        create_threshold_policy(session, name="second", levels=[level_1_level])
        assert sorted(policy.name for policy in list_threshold_policies(session)) == ["first", "second"]


def test_delete_removes_an_unused_policy(engine, level_1_level):
    with Session(engine) as session:
        policy = create_threshold_policy(session, name="p", levels=[level_1_level])
        assert policy.id is not None

        delete_threshold_policy(session, policy.id)
        assert list_threshold_policies(session) == []


def test_delete_is_refused_while_a_prediction_references_it(signal_prediction):
    session, prediction = signal_prediction
    prediction.prediction_setup.threshold_policy = None
    session.commit()

    with pytest.raises(ThresholdPolicyInUseError, match="used by prediction"):
        delete_threshold_policy(session, prediction.threshold_policy_id)


def test_delete_is_refused_while_a_setup_points_at_it(engine, level_1_level):
    """A running forecast must not lose the definition of what counts as a signal."""
    with Session(engine) as session:
        policy = create_threshold_policy(session, name="p", levels=[level_1_level])
        assert policy.id is not None
        backtest_id = _make_backtest(session)
        create_prediction_setup(
            session,
            backtest_id=backtest_id,
            name="setup",
            schedule_cron_expression=None,
            schedule_enabled=False,
            quantile_targets=[],
            threshold_policy_id=policy.id,
        )

        with pytest.raises(ThresholdPolicyInUseError):
            delete_threshold_policy(session, policy.id)


def test_delete_is_refused_while_signals_reference_it(engine, level_1_level):
    """A signal's level name only means something against its policy."""
    from chap_core.database.signal_tables import Signal
    from chap_core.services.signal_service import create_signals

    with Session(engine) as session:
        policy = create_threshold_policy(session, name="p", levels=[level_1_level])
        assert policy.id is not None
        create_signals(
            session,
            [Signal(time_period="2024-07", org_unit="A", threshold_policy_id=policy.id, level="level_1")],
        )

        with pytest.raises(ThresholdPolicyInUseError, match="signals raised against it"):
            delete_threshold_policy(session, policy.id)
