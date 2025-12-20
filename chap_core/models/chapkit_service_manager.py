"""
Manages lifecycle of chapkit model services started from local directories.
"""

import logging
import os
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Optional

import httpx

from chap_core.exceptions import ChapkitServiceStartupError

logger = logging.getLogger(__name__)


def find_available_port(start_port: int = 8000, max_attempts: int = 100) -> int:
    """Find an available port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise ChapkitServiceStartupError(f"Could not find available port in range {start_port}-{start_port + max_attempts}")


def is_url(path_or_url: str) -> bool:
    """Detect if input is a URL or a directory path."""
    return path_or_url.startswith(("http://", "https://"))


class ChapkitServiceManager:
    """
    Manages the lifecycle of a chapkit model service subprocess.

    Usage:
        with ChapkitServiceManager("/path/to/model") as manager:
            url = manager.url
            # Use the service...
    """

    def __init__(
        self,
        model_directory: str,
        port: Optional[int] = None,
        host: str = "127.0.0.1",
        startup_timeout: int = 60,
    ):
        """
        Initialize the service manager.

        Args:
            model_directory: Path to the chapkit model directory
            port: Specific port to use, or None to auto-detect
            host: Host to bind to (default: 127.0.0.1)
            startup_timeout: Seconds to wait for service to become healthy
        """
        self.model_directory = Path(model_directory).resolve()
        self.host = host
        self.port = port
        self.startup_timeout = startup_timeout
        self._process: Optional[subprocess.Popen] = None
        self._url: Optional[str] = None

    @property
    def url(self) -> str:
        """Return the URL of the running service."""
        if self._url is None:
            raise RuntimeError("Service not started. Use as context manager.")
        return self._url

    def _validate_directory(self) -> None:
        """Validate that the model directory exists and is valid."""
        if not self.model_directory.exists():
            raise ChapkitServiceStartupError(f"Model directory does not exist: {self.model_directory}")
        if not self.model_directory.is_dir():
            raise ChapkitServiceStartupError(f"Model path is not a directory: {self.model_directory}")

    def _start_service(self) -> None:
        """Start the fastapi dev server as a subprocess."""
        if self.port is None:
            self.port = find_available_port()

        self._url = f"http://{self.host}:{self.port}"

        command = [
            "uv",
            "run",
            "fastapi",
            "dev",
            "--port",
            str(self.port),
            "--host",
            self.host,
        ]

        logger.info(f"Starting chapkit service at {self._url} from {self.model_directory}")

        self._process = subprocess.Popen(
            command,
            cwd=self.model_directory,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid if os.name != "nt" else None,
        )

    def _wait_for_healthy(self) -> None:
        """Wait for the service to become healthy."""
        start_time = time.time()
        health_url = f"{self._url}/health"

        while time.time() - start_time < self.startup_timeout:
            if self._process.poll() is not None:
                stdout, stderr = self._process.communicate()
                raise ChapkitServiceStartupError(
                    f"Service process died during startup.\nstdout: {stdout.decode()}\nstderr: {stderr.decode()}"
                )

            try:
                response = httpx.get(health_url, timeout=2)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == "healthy":
                        logger.info(f"Service at {self._url} is healthy")
                        return
            except (httpx.RequestError, httpx.HTTPStatusError):
                pass

            logger.debug(f"Waiting for service at {health_url}...")
            time.sleep(1)

        self._stop_service()
        raise ChapkitServiceStartupError(
            f"Service at {self._url} did not become healthy within {self.startup_timeout} seconds"
        )

    def _stop_service(self) -> None:
        """Stop the running service subprocess gracefully."""
        if self._process is None:
            return

        logger.info(f"Stopping chapkit service at {self._url}")

        try:
            if os.name != "nt":
                os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
            else:
                self._process.terminate()

            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("Service did not stop gracefully, killing...")
                if os.name != "nt":
                    os.killpg(os.getpgid(self._process.pid), signal.SIGKILL)
                else:
                    self._process.kill()
                self._process.wait()
        except ProcessLookupError:
            pass
        finally:
            self._process = None
            self._url = None

    def __enter__(self) -> "ChapkitServiceManager":
        """Start the service when entering context."""
        self._validate_directory()
        self._start_service()
        self._wait_for_healthy()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop the service when exiting context."""
        self._stop_service()
