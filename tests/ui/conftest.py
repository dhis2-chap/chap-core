import os

import pytest


@pytest.fixture(autouse=True)
def nothing_runs_on_this_machine(monkeypatch, tmp_path_factory):
    """Keep the UI tests off the network and the machine's own chaps and Docker, which can be slow or busy.

    The marketplace is unreachable, a chaps that lists nothing goes first on PATH, and no Docker
    container runs. Tests that need something else set their own.
    """
    bin_dir = tmp_path_factory.mktemp("bin")
    chaps = bin_dir / "chaps"
    chaps.write_text("#!/bin/sh\ncase \"$*\" in\n  *' ps') echo '{\"models\": []}' ;;\n  *) echo '[]' ;;\nesac\n")
    chaps.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setattr("chap_core.ui.models.list_services", lambda images=None: [])
    monkeypatch.setattr("chap_core.services.model_marketplace.list_models", lambda *args, **kwargs: ([], {}))
