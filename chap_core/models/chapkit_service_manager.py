"""
Manages lifecycle of chapkit model services started from local directories.
"""

import collections
import logging
import os
import re
import shutil
import signal
import socket
import subprocess
import threading
import time
from collections.abc import Sequence
from pathlib import Path

import httpx

from chap_core.exceptions import ChapkitServicePortInUseError, ChapkitServiceStartupError

logger = logging.getLogger(__name__)

# Number of most recent service output lines kept for error messages.
OUTPUT_TAIL_LINES = 200

# ANSI colour and style codes, stripped from service output before matching.
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")

# Model source prefix for a chapkit model run from its container image, as in docker://ghcr.io/org/model:tag.
DOCKER_SOURCE_PREFIX = "docker://"

# The port chapkit images serve on inside the container.
CONTAINER_PORT = 8000

# Startup attempts when an auto-selected port is taken between selection and bind,
# which happens when several services are started at the same time.
PORT_ATTEMPTS = 5


def find_available_port(start_port: int = 8001, max_attempts: int = 99) -> int:
    """Find an available port starting from start_port.

    The default range is 8001-8099; 8000 is skipped because it is often taken
    (by chap itself, among others).
    """
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise ChapkitServiceStartupError(
        f"Could not find available port in range {start_port}-{start_port + max_attempts - 1}"
    )


def is_url(path_or_url: str | Path) -> bool:
    """Detect if input is a URL or a directory path."""
    return str(path_or_url).startswith(("http://", "https://"))


class ChapkitServiceManager:
    """
    Manages the lifecycle of a chapkit model service subprocess.

    With ``require_listening_line``, the service is only accepted once its own
    output reports that it is listening on the selected port (uvicorn's
    "Uvicorn running on <url>" line). A healthy response alone could come from
    another service that bound the same port first. This needs a service whose
    log output is known, such as ``chapkit mlproject run``; without it, a healthy
    ``/health`` response is enough.

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
        command: Sequence[str] | None = None,
        env: dict[str, str] | None = None,
        require_listening_line: bool = False,
    ):
        """
        Initialize the service manager.

        Args:
            model_directory: Path to the chapkit model directory
            port: Specific port to use, or None to auto-detect
            host: Host to bind to (default: 127.0.0.1)
            startup_timeout: Seconds to wait for service to become healthy
            command: Command that starts the service, without the --port and --host
                options, which are appended (default: uv run fastapi dev)
            env: Environment for the service process (default: inherit the current one)
            require_listening_line: Only accept the service after its own output reports
                uvicorn's "Uvicorn running on <url>" line
        """
        self.model_directory = Path(model_directory).resolve()
        self.host = host
        self.port = port
        # The port the caller asked for; None means pick a free one on every start.
        self._requested_port = port
        self.startup_timeout = startup_timeout
        self.command = list(command) if command is not None else ["uv", "run", "fastapi", "dev"]
        self.env = env
        self.require_listening_line = require_listening_line
        self._process: subprocess.Popen | None = None
        self._url: str | None = None
        self._output: collections.deque[str] = collections.deque(maxlen=OUTPUT_TAIL_LINES)
        self._reader: threading.Thread | None = None
        self._listening = threading.Event()

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
        """Start the service as a subprocess."""
        if self.port is None:
            self.port = find_available_port()

        self._url = f"http://{self.host}:{self.port}"

        command = self._launch_command()

        logger.info(f"Starting chapkit service at {self._url} from {self._source()}")

        self._output.clear()
        # A fresh event per launch, so a previous process's reader cannot mark this one as listening.
        self._listening = threading.Event()
        self._process = subprocess.Popen(
            command,
            cwd=self.model_directory,
            env=self.env,
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
            # The lookahead keeps port 800 from matching a line about port 8000.
            args=(self._process.stdout, re.compile(rf"running on {re.escape(self._url)}(?!\d)"), self._listening),
            name=f"chapkit-service-output-{self.port}",
            daemon=True,
        )
        self._reader.start()

    def _launch_command(self) -> list[str]:
        """The command that starts the service on ``self.port``."""
        return [*self.command, "--port", str(self.port), "--host", self.host]

    def _source(self) -> str:
        """What the service is started from, for log messages."""
        return str(self.model_directory)

    def _prepare(self) -> None:
        """Check what the service is started from, before the first start."""
        self._validate_directory()

    def _pump_output(self, stream, listening_marker: re.Pattern[str], listening: threading.Event) -> None:
        """Drain the service's merged output until EOF, keeping a bounded tail.

        Sets ``listening`` when the service reports that it is serving on its URL.
        """
        try:
            for line in stream:
                line = line.rstrip("\n")
                if listening_marker.search(ANSI_ESCAPE.sub("", line).lower()):
                    listening.set()
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
                message = (
                    f"Service process died during startup with exit code {self._process.returncode}.\n"
                    f"Recent output:\n{self.recent_output()}"
                )
                output = self.recent_output().lower()
                # A process reports "address already in use"; docker reports "port is already allocated".
                if "already in use" in output or "already allocated" in output:
                    raise ChapkitServicePortInUseError(message)
                raise ChapkitServiceStartupError(message)

            if self.require_listening_line and not self._listening.is_set():
                logger.debug(f"Waiting for the service to report it is listening on {self._url}...")
                time.sleep(0.2)
                continue

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
        listening = self._listening.is_set()
        self._stop_service()
        reason = (
            "did not become healthy"
            if listening or not self.require_listening_line
            else f"never reported that it was listening (no 'Uvicorn running on {url}' line)"
        )
        raise ChapkitServiceStartupError(
            f"Service at {url} {reason} within {self.startup_timeout} seconds.\nRecent output:\n{self.recent_output()}"
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
        """Start the service when entering context.

        An auto-selected port is only probed, not held, so another process can
        bind it before the service does. In that case the service exits with an
        "already in use" error and is started again on a newly selected port.
        Only a service that exits with that error is retried, never a timeout.
        """
        self._prepare()
        self.port = self._requested_port
        auto_port = self._requested_port is None
        for attempt in range(1, PORT_ATTEMPTS + 1):
            self._start_service()
            try:
                self._wait_for_healthy()
                return self
            except ChapkitServicePortInUseError:
                if not (auto_port and attempt < PORT_ATTEMPTS):
                    raise
                logger.info(f"Port {self.port} was taken before the service could bind it, retrying on another port")
                self._stop_service()
                self.port = None
            except BaseException:
                # __exit__ does not run when __enter__ fails, so a service interrupted while
                # starting (Ctrl-C, a closed terminal) is stopped here.
                self._stop_service()
                raise
        raise AssertionError("unreachable")

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop the service when exiting context."""
        self._stop_service()


def ensure_docker_image(image: str) -> str:
    """Pull ``image`` unless it is present, and return the platform to run it with, such as ``linux/amd64``.

    An image with no build for this machine, such as an amd64-only image on Apple Silicon, is pulled
    for linux/amd64 and runs under emulation.
    """
    if shutil.which("docker") is None:
        raise ChapkitServiceStartupError(f"Running {image} needs Docker, but the docker command was not found.")

    def platform() -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["docker", "image", "inspect", "--format", "{{.Os}}/{{.Architecture}}", image],
            capture_output=True,
            text=True,
            check=False,
        )

    inspected = platform()
    if inspected.returncode != 0:
        logger.info(f"Pulling {image}")
        pulled = subprocess.run(["docker", "pull", "-q", image], capture_output=True, text=True, check=False)
        if pulled.returncode != 0 and "no matching manifest" in pulled.stderr.lower():
            logger.info(f"{image} has no build for this machine; pulling linux/amd64, which runs under emulation")
            pulled = subprocess.run(
                ["docker", "pull", "-q", "--platform", "linux/amd64", image],
                capture_output=True,
                text=True,
                check=False,
            )
        if pulled.returncode != 0:
            raise ChapkitServiceStartupError(f"Could not pull {image}: {pulled.stderr.strip()}")
        inspected = platform()
    return inspected.stdout.strip()


class DockerChapkitServiceManager(ChapkitServiceManager):
    """Runs a chapkit model's container image for the duration of a run.

    The container is started with ``docker run --rm`` on a free local port and removed when the
    service stops, also if it outlived the docker process.
    """

    def __init__(self, image: str, port: int | None = None, host: str = "127.0.0.1", startup_timeout: int = 300):
        """
        Args:
            image: Container image, such as ``ghcr.io/chap-models/chapkit_ghr_model:sha-84fbe77``
            port: Local port to publish the service on, or None to pick a free one
            host: Local address to publish the service on
            startup_timeout: Seconds to wait for the service to become healthy. Images that run under
                emulation can take minutes to start.
        """
        super().__init__(model_directory=".", port=port, host=host, startup_timeout=startup_timeout)
        self.image = image
        self._platform: str | None = None
        self._container: str | None = None

    def _source(self) -> str:
        return self.image

    def _prepare(self) -> None:
        self._platform = ensure_docker_image(self.image)

    def _launch_command(self) -> list[str]:
        name = re.sub(r"[^a-zA-Z0-9_.-]+", "-", self.image.rsplit("/", 1)[-1])
        self._container = f"chap-{name}-{self.port}"
        platform = ["--platform", self._platform] if self._platform else []
        return [
            "docker",
            "run",
            "--rm",
            "--name",
            self._container,
            *platform,
            "-p",
            f"{self.host}:{self.port}:{CONTAINER_PORT}",
            self.image,
        ]

    def _stop_service(self) -> None:
        super()._stop_service()
        if self._container is not None:
            subprocess.run(["docker", "rm", "-f", self._container], capture_output=True, check=False)
            self._container = None
