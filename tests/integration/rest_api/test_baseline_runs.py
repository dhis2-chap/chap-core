import dataclasses

import pytest
from fastapi.testclient import TestClient
from sqlmodel import select

from chap_core.database.database import SessionWrapper
from chap_core.database.dataset_tables import DataSet
from chap_core.database.model_templates_and_config_tables import ModelConfiguration
from chap_core.database.tables import Backtest, SpecificationBaseline, SpecificationBaselineStatus
from chap_core.models.builtin import builtin_template_config, get_builtin_model
from chap_core.rest_api.app import app
from chap_core.rest_api.baseline_runs import request_baselines, run_baseline
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
        raise ValueError("too little history")

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


def test_reading_a_specification_queues_its_missing_baselines(override_session, specification_id, baseline_worker):
    assert client.get(f"/v1/crud/backtest-specifications/{specification_id}").status_code == 200
    assert len(baseline_worker.queued) == 3
    assert client.get(f"/v1/crud/backtest-specifications/{specification_id}").status_code == 200
    assert len(baseline_worker.queued) == 3
