import runpy
import sys
from pathlib import Path

import pytest

if sys.platform == "win32":
    pytest.skip("gunicorn imports Unix-only modules (fcntl, pwd, grp)", allow_module_level=True)

from gunicorn.config import Config  # noqa: E402

GUNICORN_CONF = Path(__file__).parent.parent / "gunicorn.conf.py"


def _load_config() -> Config:
    config = Config()
    for name, value in runpy.run_path(str(GUNICORN_CONF)).items():
        if name in config.settings:
            config.set(name, value)
    return config


def test_gunicorn_conf_disables_control_socket():
    assert _load_config().control_socket_disable is True


def test_post_fork_gives_the_worker_its_own_connections(clean_engine, monkeypatch):
    """A forked worker gets a fresh pool; the main process's connection stays open (CLIM-1808)."""
    import chap_core.database.database as database

    monkeypatch.setattr(database, "engine", clean_engine)
    with clean_engine.connect() as connection:
        inherited = connection.connection.dbapi_connection
    inherited_pool = clean_engine.pool

    _load_config().post_fork(None, None)

    assert clean_engine.pool is not inherited_pool
    assert clean_engine.pool.checkedin() == 0
    # Not closed: in production it still belongs to the gunicorn main process.
    assert inherited.execute("SELECT 1").fetchone() == (1,)
