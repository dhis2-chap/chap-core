import os
import subprocess
import sys
from unittest.mock import patch

import pandas as pd
import pytest

from chap_core.assessment.dataset_splitting import train_test_split_with_weather
from chap_core.datatypes import ClimateHealthTimeSeries
from chap_core.file_io.example_data_set import datasets

# from chap_core.external.models.jax_models.model_spec import NutsParams
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.time_period import Month
from chap_core.datatypes import FullData


@pytest.fixture
def dataset():
    return datasets["ISIMIP_dengue_harmonized"].load()["brazil"]


@pytest.fixture()
def data(data_path):
    file_name = (data_path / "hydro_met_subset").with_suffix(".csv")
    return DataSet.from_pandas(pd.read_csv(file_name), ClimateHealthTimeSeries)


@pytest.fixture()
def train_data(split_data):
    return split_data[0]


@pytest.fixture()
def split_data(data):
    # Month accepts positional args (year, month) via *args pattern
    return train_test_split_with_weather(data, Month(2013, 4))  # pyright: ignore[reportArgumentType]


@pytest.fixture()
def test_data(split_data):
    return split_data[1:]


@pytest.fixture
def full_train_data(train_data):
    return train_data.add_fields(FullData, population=lambda data: [100000] * len(data))


# A stand-in for `uv run fastapi dev`: a tiny HTTP server that answers /health and
# writes more than a pipe buffer to stdout and stderr. Used to test
# ChapkitServiceManager without a chapkit model installed.
_FAKE_CHAPKIT_SERVICE = """
import http.server
import json
import sys
import threading
import time

mode, port = sys.argv[1], int(sys.argv[2])

if mode == "die":
    print("fake service refused to start")
    sys.stdout.flush()
    sys.exit(3)

if mode == "slow_port_in_use":
    time.sleep(2)

if mode in ("port_in_use", "slow_port_in_use"):
    print(f"Error: Port {port} on 127.0.0.1 is already in use.")
    sys.stdout.flush()
    sys.exit(1)

status = "healthy" if mode in ("flood", "invalid_bytes", "silent", "colored") else "starting"


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({"status": status}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def flood():
    chunk = "x" * 1024
    for _ in range(256):
        sys.stdout.write(chunk + "\\n")
        sys.stderr.write(chunk + "\\n")
    sys.stdout.write("STDOUT_FLOOD_DONE\\n")
    sys.stdout.flush()
    sys.stderr.write("STDERR_FLOOD_DONE\\n")
    sys.stderr.flush()


def invalid_bytes():
    sys.stdout.buffer.write(b"\\xff\\n")
    sys.stdout.buffer.flush()
    sys.stdout.write("AFTER_INVALID_BYTES\\n")
    sys.stdout.flush()


if mode == "flood":
    threading.Thread(target=flood, daemon=True).start()
if mode == "invalid_bytes":
    threading.Thread(target=invalid_bytes, daemon=True).start()

server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
if mode == "colored":
    # uvicorn's color_message: the URL in bold.
    print(f"INFO:     Uvicorn running on \x1b[1mhttp://127.0.0.1:{port}\x1b[0m (Press CTRL+C to quit)")
elif mode == "noisy_starting":
    print("a model dependency warns that some resource is already in use")
elif mode != "silent":
    print(f"INFO:     Uvicorn running on http://127.0.0.1:{port} (Press CTRL+C to quit)")
sys.stdout.flush()
server.serve_forever()
"""


@pytest.fixture
def fake_chapkit_service(tmp_path):
    """Return a factory that patches Popen in the service manager to launch the fake service.

    Each mode is one of ``"flood"`` (healthy, then floods stdout and stderr),
    ``"invalid_bytes"`` (healthy, writes a byte that is not valid UTF-8, then keeps logging),
    ``"die"`` (prints a line and exits 3), ``"port_in_use"`` (reports its port as taken and
    exits 1), ``"slow_port_in_use"`` (the same after two seconds), ``"silent"`` (healthy, but
    never prints uvicorn's "running on" line), ``"colored"`` (healthy, prints that line with a
    bold URL), ``"noisy_starting"`` (never healthy, and logs the phrase "already in use"), or
    ``"starting"`` (never becomes healthy). Given several modes, successive
    launches use them in order and the last one repeats.
    """
    script = tmp_path / "fake_chapkit_service.py"
    script.write_text(_FAKE_CHAPKIT_SERVICE)
    real_popen = subprocess.Popen

    def factory(*modes: str):
        remaining = list(modes)

        def launch(command, **kwargs):
            port = command[command.index("--port") + 1]
            mode = remaining.pop(0) if len(remaining) > 1 else remaining[0]
            return real_popen([sys.executable, str(script), mode, port], **kwargs)

        return patch("chap_core.models.chapkit_service_manager.subprocess.Popen", side_effect=launch)

    return factory


FAKE_DOCKER = """\
import os, pathlib, sys
from http.server import BaseHTTPRequestHandler, HTTPServer

state = pathlib.Path(os.environ["FAKE_DOCKER_STATE"])
args = sys.argv[1:]
with open(state / "calls.log", "a") as log:
    log.write(" ".join(args) + "\\n")
if args[:2] == ["image", "inspect"]:
    if (state / "pulled").exists():
        print("linux/amd64")
        sys.exit(0)
    sys.exit(1)
if args[0] == "pull":
    if "--platform" in args:
        (state / "pulled").touch()
        sys.exit(0)
    print("no matching manifest for linux/arm64/v8 in the manifest list entries", file=sys.stderr)
    sys.exit(1)
if args[0] == "run":
    port = int(args[args.index("-p") + 1].split(":")[1])

    class Health(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "healthy"}')

        def log_message(self, *args):
            pass

    HTTPServer(("127.0.0.1", port), Health).serve_forever()
"""


@pytest.fixture
def fake_docker(tmp_path, monkeypatch):
    """A `docker` on PATH for an amd64-only image: the first pull fails without --platform, and `run`
    serves a healthy chapkit /health endpoint on the published port. Returns the file of its calls."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text(f"#!{sys.executable}\n{FAKE_DOCKER}")
    docker.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_DOCKER_STATE", str(tmp_path))
    return tmp_path / "calls.log"
