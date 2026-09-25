"""End-to-end integration tests with a real chapkit service.

Starts a minimal chapkit model as a subprocess, verifies self-registration
with chap-core, and runs a backtest via the CLI.
"""

import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path

import httpx
import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel

import chap_core.database.tables  # noqa: F401 - register all table models
from chap_core.database.database import SessionWrapper
from chap_core.database.dataset_manager import DataSetManager
from chap_core.database.model_templates_and_config_tables import ModelConfiguration
from chap_core.database.tables import Backtest
from chap_core.models.external_chapkit_model import ml_service_info_to_model_template_config
from chap_core.rest_api.data_models import BacktestCreate
from chap_core.rest_api.db_worker_functions import run_backtest
from chap_core.rest_api.services.schemas import MLServiceInfo

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "chapkit_test_model"
# Chapkit reports the GIT_REVISION env var as git_revision. A template must be stored
# from the revision its service reports, or a run against it is refused.
FIXTURE_GIT_REVISION = "0123456789abcdef0123456789abcdef01234567"
EXAMPLE_DATA = Path(__file__).parent.parent.parent / "example_data"
EXAMPLE_CSV = EXAMPLE_DATA / "vietnam_monthly.csv"
EXAMPLE_GEOJSON = EXAMPLE_DATA / "vietnam_monthly.geojson"


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _wait_for_health(url: str, timeout: float = 60.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"{url}/health", timeout=3)
            if r.status_code == 200:
                return True
        except httpx.ConnectError:
            pass
        time.sleep(1)
    return False


@pytest.fixture(scope="module")
def chapkit_train_log(tmp_path_factory) -> Path:
    """Path the fixture service appends one JSON line to per train call."""
    return tmp_path_factory.mktemp("chapkit_train_log") / "train_calls.jsonl"


@pytest.fixture(scope="module")
def chapkit_service(tmp_path_factory, chapkit_train_log):
    """Start a real chapkit service as a subprocess."""
    port = _find_free_port()
    url = f"http://127.0.0.1:{port}"
    data_dir = tmp_path_factory.mktemp("chapkit_data")

    # Install deps on first run
    subprocess.run(
        ["uv", "sync", "--directory", str(FIXTURE_DIR)],
        check=True,
        capture_output=True,
    )

    env = {
        **os.environ,
        "SERVICEKIT_ORCHESTRATOR_URL": "",  # disable registration for CLI test
        "GIT_REVISION": FIXTURE_GIT_REVISION,
        "CHAPKIT_DATABASE_URL": f"sqlite+aiosqlite:///{data_dir}/chapkit.db",
        "CHAPKIT_TEST_TRAIN_LOG": str(chapkit_train_log),
    }

    # Log to files rather than pipes: chapkit logs every request, and an
    # unread pipe fills up, blocking the server so it can never shut down.
    stdout_path = data_dir / "stdout.log"
    stderr_path = data_dir / "stderr.log"
    stdout_file = stdout_path.open("wb")
    stderr_file = stderr_path.open("wb")

    proc = subprocess.Popen(
        [
            "uv",
            "run",
            "--directory",
            str(FIXTURE_DIR),
            "uvicorn",
            "main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        env=env,
        stdout=stdout_file,
        stderr=stderr_file,
    )

    try:
        if not _wait_for_health(url):
            stdout = stdout_path.read_text()
            stderr = stderr_path.read_text()
            pytest.fail(f"Chapkit service failed to start:\nstdout: {stdout}\nstderr: {stderr}")
        yield url
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        stdout_file.close()
        stderr_file.close()


def _in_memory_engine():
    """A fresh in-memory database with the chap-core schema created."""
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(engine)
    return engine


def _register_service_and_dataset(session: SessionWrapper, chapkit_service: str) -> int:
    """Register the live service as a model template with a default configured model.

    Returns the id of the example dataset the backtest runs on.
    """
    info_response = httpx.get(f"{chapkit_service}/api/v1/info")
    info = MLServiceInfo.model_validate(info_response.json())
    template_config = ml_service_info_to_model_template_config(info, chapkit_service)
    template_id = session.add_model_template_from_yaml_config(template_config, source_digest=info.git_revision)
    session.add_configured_model(template_id, ModelConfiguration(), uses_chapkit=True)
    return DataSetManager(session.session).save_dataset_from_csv("vietnam_test", EXAMPLE_CSV, EXAMPLE_GEOJSON)


@pytest.mark.slow
def test_chapkit_service_is_healthy(chapkit_service):
    """Verify the chapkit test model is running and responds to health checks."""
    r = httpx.get(f"{chapkit_service}/health")
    assert r.status_code == 200


@pytest.mark.slow
def test_chapkit_service_info(chapkit_service):
    """Verify the service exposes correct metadata."""
    r = httpx.get(f"{chapkit_service}/api/v1/info")
    assert r.status_code == 200
    info = r.json()
    assert info["id"] == "chapkit-test-model"
    assert info["period_type"] == "monthly"
    assert info["git_revision"] == FIXTURE_GIT_REVISION


@pytest.mark.slow
def test_chapkit_eval_cli(chapkit_service, tmp_path):
    """Run chap eval against the live chapkit service via CLI."""
    output_file = tmp_path / "chapkit_eval_output.nc"

    result = subprocess.run(
        [
            "uv",
            "run",
            "chap",
            "eval",
            "--model-name",
            chapkit_service,
            "--dataset-csv",
            str(EXAMPLE_CSV),
            "--output-file",
            str(output_file),
            "--backtest-params.n-splits",
            "2",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 0, f"chap eval failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    assert output_file.exists(), "Output file was not created"
    assert output_file.stat().st_size > 0, "Output file is empty"


@pytest.mark.slow
def test_chapkit_backtest_via_worker_function(chapkit_service):
    """Run a backtest against the live chapkit service using the DB worker function.

    This mirrors the REST API backtest flow (POST /v1/crud/backtests/) but
    calls run_backtest() directly, bypassing Celery.
    """
    engine = _in_memory_engine()

    with SessionWrapper(engine) as session:
        dataset_id = _register_service_and_dataset(session, chapkit_service)

        # Run backtest directly (bypasses Celery)
        backtest_id = run_backtest(
            BacktestCreate(dataset_id=dataset_id, model_id="chapkit-test-model"),
            n_splits=2,
            session=session,
        )

        assert backtest_id is not None
        assert isinstance(backtest_id, int)

        # Regression: run_backtest should populate aggregate_metrics on the
        # backtest row so the `GET /v1/crud/backtests/{id}/full` response
        # includes global CRPS/MAPE/RMSE/etc without needing a round-trip
        # through the Vega visualization endpoints.
        fetched = session.session.get(Backtest, backtest_id)
        assert fetched is not None
        assert fetched.aggregate_metrics, (
            f"expected non-empty aggregate_metrics on backtest {backtest_id}, got {fetched.aggregate_metrics!r}"
        )
        assert any(k.startswith("crps") for k in fetched.aggregate_metrics), (
            f"expected at least one CRPS variant in aggregate_metrics, got keys {list(fetched.aggregate_metrics)}"
        )


@pytest.mark.slow
def test_chapkit_train_receives_the_requested_horizon(chapkit_service, chapkit_train_log):
    """The horizon chap asks for reaches the service's train function through run_info.

    The fixture service's own config defaults to 3 periods, so a train call that sees
    5 can only have got it from the run_info chap-core sent.
    """
    chapkit_train_log.write_text("")

    with SessionWrapper(_in_memory_engine()) as session:
        dataset_id = _register_service_and_dataset(session, chapkit_service)
        backtest_id = run_backtest(
            BacktestCreate(dataset_id=dataset_id, model_id="chapkit-test-model"),
            n_periods=5,
            n_splits=2,
            session=session,
        )

    assert backtest_id is not None
    logged = [json.loads(line) for line in chapkit_train_log.read_text().splitlines() if line.strip()]
    assert logged, "the service recorded no train calls"
    assert all(call["prediction_periods"] == 5 for call in logged), logged
