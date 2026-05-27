"""End-to-end integration tests with a real chapkit service.

Starts a minimal chapkit model as a subprocess, verifies self-registration
with chap-core, and runs a backtest via the CLI.
"""

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
from chap_core.database.model_templates_and_config_tables import ModelConfiguration
from chap_core.database.tables import Backtest
from chap_core.models.external_chapkit_model import ml_service_info_to_model_template_config
from chap_core.rest_api.data_models import BacktestCreate
from chap_core.rest_api.db_worker_functions import run_backtest
from chap_core.rest_api.services.schemas import MLServiceInfo

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "chapkit_test_model"
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
def chapkit_service(tmp_path_factory):
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
        "CHAPKIT_DATABASE_URL": f"sqlite+aiosqlite:///{data_dir}/chapkit.db",
    }

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
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        if not _wait_for_health(url):
            stdout = proc.stdout.read().decode() if proc.stdout else ""
            stderr = proc.stderr.read().decode() if proc.stderr else ""
            pytest.fail(f"Chapkit service failed to start:\nstdout: {stdout}\nstderr: {stderr}")
        yield url
    finally:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=10)


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
    # Set up in-memory database
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SQLModel.metadata.create_all(engine)

    with SessionWrapper(engine) as session:
        # Fetch service info and register as model template
        info_response = httpx.get(f"{chapkit_service}/api/v1/info")
        info = MLServiceInfo.model_validate(info_response.json())
        template_config = ml_service_info_to_model_template_config(info, chapkit_service)
        template_id = session.add_model_template_from_yaml_config(template_config)

        # Create configured model (default config, uses_chapkit=True)
        session.add_configured_model(template_id, ModelConfiguration(), uses_chapkit=True)

        # Add dataset from example CSV
        dataset_id = session.add_dataset_from_csv("vietnam_test", EXAMPLE_CSV, EXAMPLE_GEOJSON)

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
