"""Lock in that chap_core forces a non-interactive matplotlib backend.

Some Linux environments (e.g. Fedora) auto-select the Tk backend, which
crashes with "bit out of range 0 - FD_SETSIZE on fd_set" under high
file-descriptor counts. chap_core/__init__.py uses os.environ.setdefault
so users can still override MPLBACKEND for interactive use.
"""

import os
import subprocess
import sys

import chap_core  # noqa: F401  -- import for side effect


def test_mplbackend_default_after_import():
    assert os.environ.get("MPLBACKEND") == "Agg"


def test_mplbackend_override_is_respected():
    env = {k: v for k, v in os.environ.items() if k != "MPLBACKEND"}
    env["MPLBACKEND"] = "Qt5Agg"
    result = subprocess.run(
        [sys.executable, "-c", "import os; import chap_core; print(os.environ['MPLBACKEND'])"],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    assert result.stdout.strip() == "Qt5Agg"
