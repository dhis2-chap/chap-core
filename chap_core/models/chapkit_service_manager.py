"""
Manages lifecycle of chapkit model services started from local directories.
"""

import collections
import logging
import os
import signal
import socket
import subprocess
import threading
import time
from pathlib import Path

import httpx

from chap_core.exceptions import ChapkitServiceStartupError

logger = logging.getLogger(__name__)

# Number of most recent service output lines kept for error messages.
OUTPUT_TAIL_LINES = 200


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


def is_url(path_or_url: str | Path) -> bool:
    """Detect if input is a URL or a directory path."""
    return str(path_or_url).startswith(("http://", "https://"))


class ChapkitServiceManager:
    """
    Manages the lifecycle of a chapkit model service subprocess.

    The service's stdout and stderr are merged into one pipe that is drained
    continuously on a background thread. Without that, a service that logs more
    than the pipe buffer holds (about 64 KiB) blocks on its next write and stops
    answering requests. Each line is forwarded to this module's logger at DEBUG
    level, and the most recent lines are kept for error messages.

    Usage:
        with ChapkitServiceManager("/path/to/model") as manager:
            url = manager.url
            # Use the service...
    """

    def __init__(
        self,
        model_directory: str,
        port: int | None = None,
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
        self._process: subprocess.Popen | None = None
        self._url: str | None = None
        self._output: collections.deque[str] = collections.deque(maxlen=OUTPUT_TAIL_LINES)
        self._reader: threading.Thread | None = None

    @property
    def url(self) -> str:
        """Return the URL of the running service."""
        if self._url is None:
            raise RuntimeError("Service not started. Use as context manager.")
        return self._url

    def recent_output(self) -> str:
        """Return the most recent lines the service wrote to stdout or stderr."""
        return "\n".join(self._output)

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

        self._output.clear()
        self._process = subprocess.Popen(
            command,
            cwd=self.model_directory,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            preexec_fn=os.setsid if os.name != "nt" else None,
        )
        self._reader = threading.Thread(
            target=self._pump_output,
            args=(self._process.stdout,),
            name=f"chapkit-service-output-{self.port}",
            daemon=True,
        )
        self._reader.start()

    def _pump_output(self, stream) -> None:
        """Drain the service's merged output until EOF, keeping a bounded tail."""
        try:
            for line in stream:
                line = line.rstrip("\n")
                self._output.append(line)
                logger.debug("[chapkit service] %s", line)
        finally:
            stream.close()

    def _join_reader(self, timeout: float) -> None:
        """Wait briefly for the output thread to finish after the process exited."""
        if self._reader is not None:
            self._reader.join(timeout)
            self._reader = None

    def _wait_for_healthy(self) -> None:
        """Wait for the service to become healthy."""
        start_time = time.time()
        health_url = f"{self._url}/health"

        while time.time() - start_time < self.startup_timeout:
            assert self._process is not None
            if self._process.poll() is not None:
                self._join_reader(timeout=2)
                raise ChapkitServiceStartupError(
                    f"Service process died during startup with exit code {self._process.returncode}.\n"
                    f"Recent output:\n{self.recent_output()}"
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

        url = self._url
        self._stop_service()
        raise ChapkitServiceStartupError(
            f"Service at {url} did not become healthy within {self.startup_timeout} seconds.\n"
            f"Recent output:\n{self.recent_output()}"
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
            self._join_reader(timeout=5)
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
