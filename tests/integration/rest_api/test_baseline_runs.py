import dataclasses
import datetime

import pytest
from fastapi.testclient import TestClient
from sqlmodel import select

from chap_core.assessment.metrics import get_comparison_description
from chap_core.database.database import SessionWrapper
from chap_core.database.dataset_tables import DataSet
from chap_core.database.model_templates_and_config_tables import ModelConfiguration, ModelTemplateRole
from chap_core.database.tables import Backtest, SpecificationBaseline, SpecificationBaselineStatus
from chap_core.models.builtin import BaselineNotApplicable, builtin_template_config, get_builtin_model
from chap_core.rest_api.app import app
from chap_core.rest_api.baseline_runs import STALE_CLAIM_AFTER, request_baselines, run_baseline
from chap_core.rest_api.data_models import BacktestCreate
from chap_core.rest_api.db_worker_functions import run_backtest

client = TestClient(app)
PARAMS = {"n_periods": 3, "n_splits": 2, "stride": 1}


@pytest.fixture
def specification_id(p_seeded_engine):
    with SessionWrapper(p_seeded_engine) as session:
        dataset_id = session.session.exec(select(DataSet.id)).first()
        backtest_id = run_backtest(
            BacktestCreate(name="model", dataset_id=dataset_id, model_id="naive_model"), session=session, **PARAMS
        )
        return session.session.get(Backtest, backtest_id).specification_id


def _run_queued(engine, worker):
    with SessionWrapper(engine) as session:
        for func, (specification_baseline_id,), _ in list(worker.queued):
            assert func is run_baseline
            run_baseline(specification_baseline_id, session=session)


def _rows(engine, specification_id):
    with SessionWrapper(engine) as session:
        rows = session.session.exec(
            select(SpecificationBaseline).where(SpecificationBaseline.specification_id == specification_id)
        ).all()
        return {session.get_configured_model_by_id_or_name(row.configured_model_id).name: row for row in rows}


def test_each_baseline_is_queued_once_per_specification(p_seeded_engine, specification_id, baseline_worker):
    with SessionWrapper(p_seeded_engine) as session:
        baselines = {model.name for model in session.get_baseline_models()}
        first = request_baselines(session, specification_id, "db-url")
        assert request_baselines(session, specification_id, "db-url") == []

    assert baselines == {"global_median", "seasonal_median", "persistence"}
    assert len(first) == len(baseline_worker.queued) == len(baselines)
    assert {kwargs["database_url"] for _, _, kwargs in baseline_worker.queued} == {"db-url"}
    rows = _rows(p_seeded_engine, specification_id)
    assert set(rows) == baselines
    assert {row.status for row in rows.values()} == {SpecificationBaselineStatus.pending}


def test_baseline_runs_store_their_backtests(p_seeded_engine, specification_id, baseline_worker):
    with SessionWrapper(p_seeded_engine) as session:
        request_baselines(session, specification_id, "db-url")
    _run_queued(p_seeded_engine, baseline_worker)

    rows = _rows(p_seeded_engine, specification_id)
    with SessionWrapper(p_seeded_engine) as session:
        for name, row in rows.items():
            assert row.status == SpecificationBaselineStatus.done
            backtest = session.session.get(Backtest, row.backtest_id)
            assert (backtest.model_id, backtest.specification_id) == (name, specification_id)
    # Running baselines does not queue more baselines.
    assert len(baseline_worker.queued) == len(rows)


def test_a_failing_baseline_is_not_applicable_and_the_others_still_run(
    p_seeded_engine, specification_id, baseline_worker, monkeypatch
):
    def fail(*args, **kwargs):
        raise BaselineNotApplicable("too little history")

    monkeypatch.setattr(get_builtin_model("persistence"), "predict", fail)
    with SessionWrapper(p_seeded_engine) as session:
        request_baselines(session, specification_id, "db-url")
    _run_queued(p_seeded_engine, baseline_worker)

    rows = _rows(p_seeded_engine, specification_id)
    assert (rows["persistence"].status, rows["persistence"].backtest_id) == (
        SpecificationBaselineStatus.not_applicable,
        None,
    )
    assert rows["persistence"].message == "too little history"
    assert rows["global_median"].status == SpecificationBaselineStatus.done
    # Not applicable is final, so reading the specification again does not retry it.
    with SessionWrapper(p_seeded_engine) as session:
        assert request_baselines(session, specification_id, "db-url") == []


def test_a_baseline_failing_for_another_reason_fails_its_job_and_is_retried_a_day_later(
    p_seeded_engine, specification_id, baseline_worker, monkeypatch
):
    # A plain ValueError may come from the shared backtest path, so it is not taken as "not applicable".
    def fail(*args, **kwargs):
        raise ValueError("bug in a shared path")

    monkeypatch.setattr(get_builtin_model("persistence"), "predict", fail)
    with SessionWrapper(p_seeded_engine) as session:
        request_baselines(session, specification_id, "db-url")
    with pytest.raises(ValueError, match="bug in a shared path"):
        _run_queued(p_seeded_engine, baseline_worker)

    failed = _rows(p_seeded_engine, specification_id)["persistence"]
    assert (failed.status, failed.message) == (SpecificationBaselineStatus.failed, "bug in a shared path")
    with SessionWrapper(p_seeded_engine) as session:
        # A failure that repeats must not queue a job on every trigger.
        assert request_baselines(session, specification_id, "db-url") == []
        row = session.session.get(SpecificationBaseline, failed.id)
        row.claimed_at = datetime.datetime.now() - STALE_CLAIM_AFTER - datetime.timedelta(minutes=1)
        session.session.commit()
        assert request_baselines(session, specification_id, "db-url") == [failed.id]
        assert session.session.get(SpecificationBaseline, failed.id).status == SpecificationBaselineStatus.pending


def test_a_baseline_still_pending_after_a_day_is_queued_again(p_seeded_engine, specification_id, baseline_worker):
    with SessionWrapper(p_seeded_engine) as session:
        first = request_baselines(session, specification_id, "db-url")
        lost = session.session.get(SpecificationBaseline, first[0])
        lost.claimed_at = datetime.datetime.now() - STALE_CLAIM_AFTER - datetime.timedelta(minutes=1)
        session.session.commit()

        assert request_baselines(session, specification_id, "db-url") == [lost.id]
        assert request_baselines(session, specification_id, "db-url") == []
    assert [args for _, args, _ in baseline_worker.queued[len(first) :]] == [(lost.id,)]


def test_a_new_baseline_version_is_queued_on_an_existing_specification(
    p_seeded_engine, specification_id, baseline_worker
):
    with SessionWrapper(p_seeded_engine) as session:
        request_baselines(session, specification_id, "db-url")
        spec = dataclasses.replace(get_builtin_model("global_median").spec, version="2")
        template_id = session.add_model_template_from_yaml_config(builtin_template_config(spec))
        session.get_model_template(template_id).role = spec.role
        model_id = session.add_configured_model(
            template_id, ModelConfiguration(additional_continuous_covariates=[], user_option_values={})
        )
        new = request_baselines(session, specification_id, "db-url")

        assert [session.session.get(SpecificationBaseline, id).configured_model_id for id in new] == [model_id]


def test_a_baseline_that_cannot_be_queued_is_not_claimed(p_seeded_engine, specification_id, monkeypatch):
    from chap_core.rest_api import baseline_runs

    class _BrokerDown:
        def queue_db(self, *args, **kwargs):
            raise ConnectionError("broker down")

    monkeypatch.setattr(baseline_runs, "worker", _BrokerDown())
    with SessionWrapper(p_seeded_engine) as session:
        assert request_baselines(session, specification_id, "db-url") == []
    assert _rows(p_seeded_engine, specification_id) == {}


def test_a_backtest_from_the_api_queues_the_baselines_of_its_specification(p_seeded_engine, baseline_worker):
    with SessionWrapper(p_seeded_engine) as session:
        dataset_id = session.session.exec(select(DataSet.id)).first()
        backtest_id = run_backtest(
            BacktestCreate(name="model", dataset_id=dataset_id, model_id="naive_model"),
            session=session,
            baseline_database_url="db-url",
            **PARAMS,
        )
        specification_id = session.session.get(Backtest, backtest_id).specification_id

    assert len(baseline_worker.queued) == len(_rows(p_seeded_engine, specification_id)) == 3


def test_a_backtest_from_the_api_runs_even_when_requesting_baselines_fails(p_seeded_engine, monkeypatch):
    from chap_core.rest_api import baseline_runs

    def fail(*args, **kwargs):
        raise RuntimeError("database hiccup")

    monkeypatch.setattr(baseline_runs, "request_baselines", fail)
    with SessionWrapper(p_seeded_engine) as session:
        dataset_id = session.session.exec(select(DataSet.id)).first()
        backtest_id = run_backtest(
            BacktestCreate(name="model", dataset_id=dataset_id, model_id="naive_model"),
            session=session,
            baseline_database_url="db-url",
            **PARAMS,
        )
        assert session.session.get(Backtest, backtest_id).forecasts


def test_reading_a_specification_queues_its_missing_baselines(override_session, specification_id, baseline_worker):
    assert client.get(f"/v1/crud/backtest-specifications/{specification_id}").status_code == 200
    assert len(baseline_worker.queued) == 3
    assert client.get(f"/v1/crud/backtest-specifications/{specification_id}").status_code == 200
    assert len(baseline_worker.queued) == 3


def _leaderboard(specification_id, **params):
    response = client.get(f"/v1/crud/backtest-specifications/{specification_id}", params=params)
    assert response.status_code == 200
    return response.json()


def test_the_leaderboard_lists_baselines_as_pending_until_they_have_run(
    override_session, specification_id, baseline_worker
):
    leaderboard = _leaderboard(specification_id)

    assert {b["configuredModel"]["name"]: b["status"] for b in leaderboard["baselines"]} == {
        "global_median": "pending",
        "seasonal_median": "pending",
        "persistence": "pending",
    }
    assert leaderboard["comparisonModels"] is None
    assert all(backtest["comparisons"] == [] for backtest in leaderboard["backtests"])
    metrics = {metric["id"]: metric for metric in leaderboard["metrics"]}
    assert (metrics["mae"]["comparisonOp"], metrics["mae"]["comparisonDescription"]) == (
        "skill_ratio",
        get_comparison_description("skill_ratio"),
    )


def test_the_leaderboard_compares_each_backtest_with_the_baselines(
    override_session, p_seeded_engine, specification_id, baseline_worker, monkeypatch
):
    def fail(*args, **kwargs):
        raise BaselineNotApplicable("too little history")

    monkeypatch.setattr(get_builtin_model("persistence"), "predict", fail)
    with SessionWrapper(p_seeded_engine) as session:
        request_baselines(session, specification_id, "db-url")
    _run_queued(p_seeded_engine, baseline_worker)
    # The seeded dataset is too small to score, so the scores are set here.
    with SessionWrapper(p_seeded_engine) as session:
        for row in _rows(p_seeded_engine, specification_id).values():
            if row.backtest_id is not None:
                session.session.get(Backtest, row.backtest_id).aggregate_metrics = {"mae": 4.0}
        model_backtest = session.session.exec(
            select(Backtest).where(Backtest.specification_id == specification_id, Backtest.model_id == "naive_model")
        ).first()
        model_backtest.aggregate_metrics = {"mae": 3.0}
        session.session.commit()
        model_backtest_id = model_backtest.id

    leaderboard = _leaderboard(specification_id)
    baselines = {b["configuredModel"]["name"]: b for b in leaderboard["baselines"]}
    assert (baselines["persistence"]["status"], baselines["persistence"]["message"]) == (
        "not_applicable",
        "too little history",
    )
    assert baselines["persistence"]["aggregateMetrics"] == {}
    done_ids = {b["configuredModel"]["id"] for b in baselines.values() if b["status"] == "done"}
    assert len(done_ids) == 2
    backtest = next(b for b in leaderboard["backtests"] if b["id"] == model_backtest_id)
    assert {c["configuredModelId"]: c["values"] for c in backtest["comparisons"]} == {
        model_id: {"mae": pytest.approx(0.25)} for model_id in done_ids
    }


def test_the_leaderboard_lists_comparison_models_on_request(override_session, p_seeded_engine, specification_id):
    with SessionWrapper(p_seeded_engine) as session:
        dataset_id = session.session.exec(select(DataSet.id)).first()
        newest_id = run_backtest(
            BacktestCreate(name="again", dataset_id=dataset_id, model_id="naive_model"), session=session, **PARAMS
        )
        newest = session.session.get(Backtest, newest_id)
        newest.configured_model.model_template.role = ModelTemplateRole.comparison
        session.session.commit()

    assert _leaderboard(specification_id)["comparisonModels"] is None
    leaderboard = _leaderboard(specification_id, includeComparisonModels=True)
    (comparison_model,) = leaderboard["comparisonModels"]
    assert (comparison_model["status"], comparison_model["backtestId"]) == ("done", newest_id)
    comparison_model_id = comparison_model["configuredModel"]["id"]
    # A backtest is never compared with a run of its own model, including an older one.
    own_runs = [b for b in leaderboard["backtests"] if b["configuredModel"]["id"] == comparison_model_id]
    others = [b for b in leaderboard["backtests"] if b["configuredModel"]["id"] != comparison_model_id]
    assert len(own_runs) == 2 and others
    assert all(b["comparisons"] == [] for b in own_runs)
    assert all([c["configuredModelId"] for c in b["comparisons"]] == [comparison_model_id] for b in others)


def test_the_leaderboard_leaves_out_archived_comparison_models(override_session, p_seeded_engine, specification_id):
    with SessionWrapper(p_seeded_engine) as session:
        backtest = session.session.exec(select(Backtest).where(Backtest.specification_id == specification_id)).first()
        backtest.configured_model.model_template.role = ModelTemplateRole.comparison
        backtest.configured_model.archived = True
        session.session.commit()

    assert _leaderboard(specification_id, includeComparisonModels=True)["comparisonModels"] == []


def test_the_leaderboard_leaves_out_backtests_of_archived_models(override_session, p_seeded_engine, specification_id):
    with SessionWrapper(p_seeded_engine) as session:
        backtest = session.session.exec(select(Backtest).where(Backtest.specification_id == specification_id)).first()
        backtest.configured_model.archived = True
        session.session.commit()
        archived = set(
            session.session.exec(
                select(Backtest.id).where(
                    Backtest.specification_id == specification_id, Backtest.model_db_id == backtest.model_db_id
                )
            ).all()
        )

    leaderboard = _leaderboard(specification_id)
    assert not archived & {b["id"] for b in leaderboard["backtests"]}
    (row,) = [s for s in client.get("/v1/crud/backtest-specifications").json() if s["id"] == specification_id]
    assert row["backtestCount"] == len(leaderboard["backtests"])
    # On request, and in the backtest listing, they are still shown.
    assert archived <= {b["id"] for b in _leaderboard(specification_id, includeArchivedModels=True)["backtests"]}
    assert archived <= {b["id"] for b in client.get("/v1/crud/backtests").json()}
