import subprocess
import sys

# Runs in a fresh interpreter, since the test process may already have imported pyproj.
SCRIPT = """
import subprocess
import threading

import chap_core.rest_api.app


def import_in_thread():
    import geopandas  # noqa: F401


thread = threading.Thread(target=import_in_thread)
thread.start()
thread.join()
print(subprocess.run(["echo", "ok"], capture_output=True, text=True).returncode)
"""


def test_subprocess_works_after_geopandas_is_imported_on_a_request_thread():
    result = subprocess.run([sys.executable, "-c", SCRIPT], capture_output=True, text=True, check=True)

    assert result.stdout.strip() == "0"
