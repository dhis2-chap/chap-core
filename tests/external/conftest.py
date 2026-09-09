import subprocess
import sys
from unittest.mock import patch

import pandas as pd
import pytest

from chap_core.assessment.dataset_splitting import train_test_split_with_weather
from chap_core.datatypes import ClimateHealthTimeSeries

# from chap_core.external.models.jax_models.model_spec import NutsParams
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet
from chap_core.time_period import Month
from chap_core.datatypes import FullData


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

mode, port = sys.argv[1], int(sys.argv[2])

if mode == "die":
    print("fake service refused to start")
    sys.stdout.flush()
    sys.exit(3)

status = "healthy" if mode in ("flood", "invalid_bytes") else "starting"


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

http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
"""


@pytest.fixture
def fake_chapkit_service(tmp_path):
    """Return a factory that patches Popen in the service manager to launch the fake service.

    ``mode`` is one of ``"flood"`` (healthy, then floods stdout and stderr),
    ``"invalid_bytes"`` (healthy, writes a byte that is not valid UTF-8, then keeps logging),
    ``"die"`` (prints a line and exits 3), or ``"starting"`` (never becomes healthy).
    """
    script = tmp_path / "fake_chapkit_service.py"
    script.write_text(_FAKE_CHAPKIT_SERVICE)
    real_popen = subprocess.Popen

    def factory(mode: str):
        def launch(command, **kwargs):
            port = command[command.index("--port") + 1]
            return real_popen([sys.executable, str(script), mode, port], **kwargs)

        return patch("chap_core.models.chapkit_service_manager.subprocess.Popen", side_effect=launch)

    return factory
