import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

if sys.platform == "win32":
    pytest.skip("gunicorn imports Unix-only modules (fcntl, pwd, grp)", allow_module_level=True)

from gunicorn.config import Config  # noqa: E402

GUNICORN_CONF = Path(__file__).parent.parent / "gunicorn.conf.py"
DOCKERFILE = Path(__file__).parent.parent / "Dockerfile"


def _dockerfile_cmd() -> list[str]:
    """The exec-form CMD of the Dockerfile, as the argument list Docker runs."""
    text = DOCKERFILE.read_text()
    start = text.index("CMD [")
    end = text.index("]", start)
    return json.loads(text[start + len("CMD ") : end + 1].replace("\\\n", ""))


def _load_config() -> Config:
    config = Config()
    for name, value in runpy.run_path(str(GUNICORN_CONF)).items():
        if name in config.settings:
            config.set(name, value)
    return config


def test_gunicorn_conf_disables_control_socket():
    assert _load_config().control_socket_disable is True


def test_image_command_has_no_arguments_gunicorn_ignores():
    """A flag with its value in one token is taken as a stray positional and ignored (CLIM-1809)."""
    command = _dockerfile_cmd()
    assert command[0] == "gunicorn"
    parsed = Config().parser().parse_args(command[1:])
    assert parsed.args == ["chap_core.rest_api.app:app"]


def test_web_concurrency_sets_the_workers_of_the_image_command():
    """The deployment's WEB_CONCURRENCY and PORT reach gunicorn when it runs the image's command."""
    script = (
        "import runpy, sys\n"
        "from gunicorn.config import Config\n"
        "config = Config()\n"
        "parsed = config.parser().parse_args(sys.argv[1:])\n"
        f"for name, value in runpy.run_path({str(GUNICORN_CONF)!r}).items():\n"
        "    if name in config.settings:\n"
        "        config.set(name, value)\n"
        "for name, value in vars(parsed).items():\n"
        "    if value is not None and name in config.settings:\n"
        "        config.set(name, value)\n"
        "print(config.workers, config.bind)\n"
    )
    env = {**os.environ, "WEB_CONCURRENCY": "3", "PORT": "8000"}
    result = subprocess.run(
        [sys.executable, "-c", script, *_dockerfile_cmd()[1:]],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        cwd=GUNICORN_CONF.parent,
    )
    assert result.stdout.strip().splitlines()[-1] == "3 ['0.0.0.0:8000']"


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
