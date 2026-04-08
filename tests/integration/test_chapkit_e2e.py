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

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "chapkit_test_model"
EXAMPLE_CSV = Path(__file__).parent.parent.parent / "example_data" / "vietnam_monthly.csv"


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
