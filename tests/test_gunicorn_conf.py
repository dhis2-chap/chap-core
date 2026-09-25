import runpy
import sys
from pathlib import Path

import pytest

if sys.platform == "win32":
    pytest.skip("gunicorn imports Unix-only modules (fcntl, pwd, grp)", allow_module_level=True)

from gunicorn.config import Config  # noqa: E402

GUNICORN_CONF = Path(__file__).parent.parent / "gunicorn.conf.py"


def test_gunicorn_conf_disables_control_socket():
    config = Config()
    for name, value in runpy.run_path(str(GUNICORN_CONF)).items():
        if name in config.settings:
            config.set(name, value)

    assert config.control_socket_disable is True
