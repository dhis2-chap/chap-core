"""Running the baseline models once per backtest specification.

`request_baselines` is the one entry point every trigger calls: a backtest resolving its
specification, and, after the response is sent, a read of a specification's results. It claims each missing baseline
with a `SpecificationBaseline` row and queues one job per claimed baseline, so it returns
without running any model and is safe to call as often as triggers fire.
"""

import logging
from typing import Any

import sqlalchemy
from sqlalchemy.engine import Connection, Engine
from sqlmodel import select

from chap_core.api_types import BacktestParams
from chap_core.database.database import SessionWrapper
from chap_core.database.tables import SpecificationBaseline, SpecificationBaselineStatus
from chap_core.rest_api.celery_tasks import JOB_NAME_KW, JOB_TYPE_KW, CeleryPool, JobType
from chap_core.rest_api.data_models import BacktestCreate
from chap_core.rest_api.db_worker_functions import run_backtest

logger = logging.getLogger(__name__)

worker: CeleryPool[Any] = CeleryPool()


def request_baselines(session: SessionWrapper, specification_id: int, database_url: str) -> list[int]:
    """Queue every baseline that has no run on this specification yet, and return the ids of the new runs.

    A baseline's run is looked up by its configured model, so a new baseline version is
    queued too. Queueing failures are logged rather than raised, so a trigger never fails
    because of them. The claim is then removed, so the next trigger tries again.
    """
    existing = set(
        session.session.exec(
            select(SpecificationBaseline.configured_model_id).where(
                SpecificationBaseline.specification_id == specification_id
            )
        ).all()
    )
    queued = []
    for model in session.get_baseline_models():
        if model.id in existing:
            continue
        assert model.id is not None
        baseline = SpecificationBaseline(specification_id=specification_id, configured_model_id=model.id)
        session.session.add(baseline)
        try:
            session.session.commit()
        except sqlalchemy.exc.IntegrityError:
            # Another trigger claimed this baseline between the lookup and the commit.
            session.session.rollback()
            continue
        assert baseline.id is not None
        try:
            worker.queue_db(
                run_baseline,
                baseline.id,
                database_url=database_url,
                **{JOB_TYPE_KW: JobType.BASELINE, JOB_NAME_KW: f"baseline/{model.name}"},
            )
        except Exception:
            logger.warning(
                f"Could not queue baseline {model.name!r} on specification {specification_id}", exc_info=True
            )
            session.session.delete(baseline)
            session.session.commit()
            continue
        queued.append(baseline.id)
    return queued


def request_baselines_after_response(bind: Engine | Connection, specification_id: int, database_url: str) -> None:
    """`request_baselines` for a read endpoint's background tasks, so the response never waits on the broker.

    It runs after the response is sent, when the request's session is closed, so it opens its own.
    """
    with SessionWrapper(bind) as session:
        request_baselines(session, specification_id, database_url)


def run_baseline(specification_baseline_id: int, session: SessionWrapper) -> int | None:
    """Run one claimed baseline as a backtest on its specification, and return the backtest id.

    A baseline that raises `ValueError`, which is how a model rejects data it cannot
    handle, is marked not applicable and returns None, so it does not fail the job.
    Any other error may be transient, so the claim is removed and the job fails,
    letting a later trigger run the baseline again.
    """
    baseline = session.session.get(SpecificationBaseline, specification_baseline_id)
    assert baseline is not None, f"Specification baseline {specification_baseline_id} not found"
    specification = baseline.specification
    params = {name: getattr(specification, name) for name in BacktestParams.model_fields}
    try:
        backtest_id: int = run_backtest(
            BacktestCreate(name="baseline", dataset_id=specification.dataset_id, model_id=baseline.configured_model_id),
            **params,
            target_column=specification.target_column,
            session=session,
        )
    except ValueError as e:
        logger.warning(f"Baseline run {specification_baseline_id} is not applicable: {e}")
        session.session.rollback()
        baseline.status = SpecificationBaselineStatus.not_applicable
        baseline.message = str(e)
        session.session.commit()
        return None
    except Exception:
        session.session.rollback()
        session.session.delete(baseline)
        session.session.commit()
        raise
    baseline.status = SpecificationBaselineStatus.done
    baseline.backtest_id = backtest_id
    session.session.commit()
    return backtest_id
