"""Unit tests for chap_core.services.signal_service.

Uses an in-memory SQLite engine with foreign-key enforcement enabled so the
referential guards are real, not mocked. These tests cover the service in
isolation; the HTTP-layer wiring is verified by the FastAPI integration tests.
"""

from __future__ import annotations

import datetime
from types import SimpleNamespace

import pytest
from sqlalchemy import inspect
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, SQLModel

from chap_core.assessment.thresholds.params import SeasonalParams
from chap_core.database.signal_tables import Signal, SignalReleaseStatus, ThresholdLevel
from chap_core.database.dataset_tables import DataSet
from chap_core.database.model_templates_and_config_tables import ConfiguredModelDB, ModelTemplateDB
from chap_core.database.tables import Prediction
from chap_core.services.threshold_policy_service import create_threshold_policy
from chap_core.services.signal_service import (
    SignalReleasedError,
    SignalNotFoundError,
    InvalidSignalError,
    create_signals,
    delete_signal,
    get_signal,
    list_signals,
    set_release_status,
)


@pytest.fixture
def policy_id(engine):
    """A two-tier policy to raise signals against."""
    with Session(engine) as session:
        policy = create_threshold_policy(
            session,
            name="ladder",
            levels=[
                ThresholdLevel(
                    name="level_1",
                    output_code="1",
                    threshold_params=SeasonalParams(type="seasonal", std_multiplier=1.0),
                    alarm_threshold=0.3,
                ),
                ThresholdLevel(
                    name="level_2",
                    output_code="2",
                    threshold_params=SeasonalParams(type="seasonal", std_multiplier=3.0),
                    alarm_threshold=0.8,
                ),
            ],
        )
        assert policy.id is not None
        return policy.id


@pytest.fixture
def prediction_id(engine):
    """A prediction for signals to be raised by."""
    with Session(engine) as session:
        template = ModelTemplateDB(name="tpl", version="1.0.0")
        session.add(template)
        dataset = DataSet(name="ds")
        session.add(dataset)
        session.commit()
        assert template.id is not None and dataset.id is not None
        model = ConfiguredModelDB(name="cfg", model_template_id=template.id)
        session.add(model)
        session.commit()
        assert model.id is not None

        prediction = Prediction(
            dataset_id=dataset.id,
            model_id="cfg",
            model_db_id=model.id,
            n_periods=3,
            name="pred",
            created=datetime.datetime.now(),
        )
        session.add(prediction)
        session.commit()
        assert prediction.id is not None
        return prediction.id


def _signal(
    policy_id: int,
    org_unit: str = "A",
    period: str = "2024-07",
    level: str = "level_2",
    prediction_id: int | None = None,
) -> Signal:
    return Signal(
        time_period=period, org_unit=org_unit, threshold_policy_id=policy_id, level=level, prediction_id=prediction_id
    )


def test_created_signals_start_pending(engine, policy_id):
    """A signal is not released until someone clears it."""
    with Session(engine) as session:
        (signal,) = create_signals(session, [_signal(policy_id)])
        assert signal.id is not None
        assert signal.release_status is SignalReleaseStatus.PENDING
        assert signal.reviewed_by is None
        assert signal.reviewed_at is None
        assert signal.created is not None


def test_a_level_the_policy_does_not_define_is_rejected(engine, policy_id):
    """A level name only means something against its policy."""
    with Session(engine) as session:
        with pytest.raises(InvalidSignalError, match="no level 'catastrophe'"):
            create_signals(session, [_signal(policy_id, level="catastrophe")])


def test_an_unknown_policy_is_rejected(engine):
    with Session(engine) as session:
        with pytest.raises(InvalidSignalError, match="ThresholdPolicy 99999 not found"):
            create_signals(session, [_signal(99999)])


def test_an_unknown_prediction_is_rejected(engine, policy_id):
    with Session(engine) as session:
        signal = _signal(policy_id)
        signal.prediction_id = 99999
        with pytest.raises(InvalidSignalError, match="Prediction 99999 not found"):
            create_signals(session, [signal])


def test_a_bad_entry_writes_none_of_the_batch(engine, policy_id):
    """One run raises many signals, so the batch is validated before anything lands."""
    with Session(engine) as session:
        with pytest.raises(InvalidSignalError):
            create_signals(session, [_signal(policy_id), _signal(policy_id, level="nonsense")])
        assert list_signals(session) == []


def test_a_duplicate_that_passes_the_check_fails_the_whole_batch(engine, policy_id, prediction_id, monkeypatch):
    """A concurrent request can insert the same signal after the check, leaving the constraint to refuse it."""
    with Session(engine) as session:
        create_signals(session, [_signal(policy_id, prediction_id=prediction_id)])
        batch = [
            _signal(policy_id, org_unit="other", prediction_id=prediction_id),
            _signal(policy_id, level="level_1", prediction_id=prediction_id),
        ]
        with monkeypatch.context() as patch:
            patch.setattr(session, "exec", lambda *_: SimpleNamespace(first=lambda: None))
            with pytest.raises(InvalidSignalError, match="none were recorded"):
                create_signals(session, batch)
        assert len(list_signals(session)) == 1


def test_a_prediction_raises_one_signal_per_org_unit_and_period(engine, policy_id, prediction_id):
    """Re-running a prediction must not queue the same signal twice."""
    with Session(engine) as session:
        create_signals(session, [_signal(policy_id, prediction_id=prediction_id)])
        with pytest.raises(InvalidSignalError, match="already has a signal"):
            create_signals(session, [_signal(policy_id, level="level_1", prediction_id=prediction_id)])
        assert len(list_signals(session)) == 1


def test_a_batch_with_two_signals_for_the_same_place_and_period_is_rejected(engine, policy_id, prediction_id):
    with Session(engine) as session:
        with pytest.raises(InvalidSignalError, match="already has a signal"):
            create_signals(
                session,
                [
                    _signal(policy_id, level="level_1", prediction_id=prediction_id),
                    _signal(policy_id, level="level_2", prediction_id=prediction_id),
                ],
            )
        assert list_signals(session) == []


def test_list_filters_narrow_the_result(engine, policy_id):
    with Session(engine) as session:
        create_signals(
            session,
            [_signal(policy_id, org_unit="A"), _signal(policy_id, org_unit="B")],
        )
        assert len(list_signals(session)) == 2
        assert [a.org_unit for a in list_signals(session, org_unit="B")] == ["B"]
        assert list_signals(session, threshold_policy_id=99999) == []


def test_pending_filter_is_the_review_queue(engine, policy_id):
    with Session(engine) as session:
        first, second = create_signals(session, [_signal(policy_id, org_unit="A"), _signal(policy_id, org_unit="B")])
        assert first.id is not None
        set_release_status(session, [first.id], SignalReleaseStatus.RELEASED, "knut")

        queue = list_signals(session, release_status=SignalReleaseStatus.PENDING)
        assert [a.org_unit for a in queue] == ["B"]
        released = list_signals(session, release_status=SignalReleaseStatus.RELEASED)
        assert [a.org_unit for a in released] == ["A"]


def test_releasing_records_who_and_when(engine, policy_id):
    with Session(engine) as session:
        (signal,) = create_signals(session, [_signal(policy_id)])
        assert signal.id is not None
        (reviewed,) = set_release_status(session, [signal.id], SignalReleaseStatus.RELEASED, "knut")
        assert reviewed.release_status is SignalReleaseStatus.RELEASED
        assert reviewed.reviewed_by == "knut"
        assert reviewed.reviewed_at is not None


def test_withholding_is_distinct_from_pending(engine, policy_id):
    """A withheld signal must not drift back into the queue for another release review."""
    with Session(engine) as session:
        (signal,) = create_signals(session, [_signal(policy_id)])
        assert signal.id is not None
        set_release_status(session, [signal.id], SignalReleaseStatus.WITHHELD, "knut")

        assert list_signals(session, release_status=SignalReleaseStatus.PENDING) == []
        withheld = list_signals(session, release_status=SignalReleaseStatus.WITHHELD)
        assert withheld[0].reviewed_by == "knut"


def test_a_whole_set_is_reviewed_together(engine, policy_id):
    with Session(engine) as session:
        signals = create_signals(session, [_signal(policy_id, org_unit=f"OU{i}") for i in range(3)])
        ids = [signal.id for signal in signals if signal.id is not None]
        reviewed = set_release_status(session, ids, SignalReleaseStatus.RELEASED, "knut")
        assert {signal.release_status for signal in reviewed} == {SignalReleaseStatus.RELEASED}


def test_one_unknown_id_leaves_the_whole_set_untouched(engine, policy_id):
    """The reviewer sees a 404 rather than a half-applied decision."""
    with Session(engine) as session:
        (signal,) = create_signals(session, [_signal(policy_id)])
        assert signal.id is not None
        with pytest.raises(SignalNotFoundError):
            set_release_status(session, [signal.id, 99999], SignalReleaseStatus.RELEASED, "knut")
        assert get_signal(session, signal.id).release_status is SignalReleaseStatus.PENDING


def test_reviewing_nothing_is_rejected(engine):
    with Session(engine) as session:
        with pytest.raises(InvalidSignalError, match="No signals"):
            set_release_status(session, [], SignalReleaseStatus.RELEASED, "knut")


def test_an_anonymous_review_is_rejected(engine, policy_id):
    with Session(engine) as session:
        (signal,) = create_signals(session, [_signal(policy_id)])
        assert signal.id is not None
        with pytest.raises(InvalidSignalError, match="reviewed_by is required"):
            set_release_status(session, [signal.id], SignalReleaseStatus.RELEASED, "  ")


def test_returning_an_signal_to_the_queue_clears_the_reviewer(engine, policy_id):
    """Sending it back to pending means nobody has decided, so no reviewer is recorded."""
    with Session(engine) as session:
        (signal,) = create_signals(session, [_signal(policy_id)])
        assert signal.id is not None
        set_release_status(session, [signal.id], SignalReleaseStatus.RELEASED, "knut")
        (returned,) = set_release_status(session, [signal.id], SignalReleaseStatus.PENDING, "knut")
        assert returned.release_status is SignalReleaseStatus.PENDING
        assert returned.reviewed_by is None
        assert returned.reviewed_at is None


def test_get_missing_signal_raises_not_found(engine):
    with Session(engine) as session:
        with pytest.raises(SignalNotFoundError):
            get_signal(session, 99999)


def test_deleting_an_signal_frees_its_prediction(engine, policy_id, prediction_id):
    """A bad run's signals can be removed so the run itself can be deleted."""
    with Session(engine) as session:
        (signal,) = create_signals(session, [_signal(policy_id, prediction_id=prediction_id)])
        assert signal.id is not None
        delete_signal(session, signal.id)

        session.delete(session.get(Prediction, prediction_id))
        session.commit()
        assert session.get(Prediction, prediction_id) is None


def test_a_released_signal_cannot_be_deleted(engine, policy_id):
    """It has been cleared for dissemination, so it stays on record."""
    with Session(engine) as session:
        (signal,) = create_signals(session, [_signal(policy_id)])
        assert signal.id is not None
        set_release_status(session, [signal.id], SignalReleaseStatus.RELEASED, "knut")
        with pytest.raises(SignalReleasedError):
            delete_signal(session, signal.id)
        assert get_signal(session, signal.id).release_status is SignalReleaseStatus.RELEASED


def test_deleting_a_missing_signal_raises_not_found(engine):
    with Session(engine) as session:
        with pytest.raises(SignalNotFoundError):
            delete_signal(session, 99999)


def test_the_database_refuses_to_delete_a_prediction_with_signals(engine, policy_id, prediction_id):
    """The RESTRICT is declared on the model field, not only in the migration.

    `create_db_and_tables` runs `SQLModel.metadata.create_all` before Alembic, so on a
    fresh database the table comes from the model and a rule declared only in the
    migration would never take effect. This asserts against a schema built by
    `create_all` with foreign keys enforced, which is what production gets.
    """
    with Session(engine) as session:
        prediction = session.get(Prediction, prediction_id)
        create_signals(session, [_signal(policy_id, prediction_id=prediction_id)])

        session.delete(prediction)
        with pytest.raises(IntegrityError):
            session.commit()
        session.rollback()


def test_the_review_queue_columns_are_indexed(engine):
    """The indexes are declared on the model, since the migration skips a table `create_all` already made."""
    indexed = {tuple(index["column_names"]) for index in inspect(engine).get_indexes("signal")}
    assert ("threshold_policy_id",) in indexed
    assert ("release_status",) in indexed
