import os
import signal
import subprocess
import sys
import time

import httpx
import pytest

from chap_core.models.chapkit_service_manager import DockerChapkitServiceManager
from chap_core.models.external_chapkit_model import ExternalChapkitModelTemplate
from chap_core.models.utils import get_model_template_from_directory_or_github_url

IMAGE = "ghcr.io/chap-models/some_model:sha-1234567"

pytestmark = pytest.mark.skipif(os.name == "nt", reason="the fake docker is a POSIX executable")


def test_image_is_pulled_run_and_removed(fake_docker):
    with DockerChapkitServiceManager(IMAGE, startup_timeout=15) as manager:
        assert httpx.get(f"{manager.url}/health").json() == {"status": "healthy"}
    calls = fake_docker.read_text().splitlines()
    assert f"pull -q {IMAGE}" in calls
    assert f"pull -q --platform linux/amd64 {IMAGE}" in calls
    run = next(call for call in calls if call.startswith("run "))
    assert "--rm" in run
    assert "--platform linux/amd64" in run
    container = run.split("--name ")[1].split()[0]
    assert calls[-1] == f"rm -f {container}"


def test_docker_source_gives_a_chapkit_template_that_runs_the_image(fake_docker):
    template = get_model_template_from_directory_or_github_url(f"docker://{IMAGE}")
    assert isinstance(template, ExternalChapkitModelTemplate)
    assert not template.is_url_mode
    assert not fake_docker.exists()


def test_docker_source_cannot_be_served_as_chapkit():
    with pytest.raises(ValueError, match="already a chapkit service"):
        get_model_template_from_directory_or_github_url(f"docker://{IMAGE}", as_chapkit=True)


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGHUP], ids=["kill", "closed terminal"])
def test_a_command_stopped_while_the_service_starts_removes_the_container(fake_docker, monkeypatch, signum):
    monkeypatch.setenv("FAKE_DOCKER_STARTUP_SECONDS", "60")
    command = [sys.executable, "-m", "chap_core.cli", "model", "schema", f"docker://{IMAGE}"]
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.monotonic() + 60
    while not (fake_docker.exists() and "run " in fake_docker.read_text()):
        assert time.monotonic() < deadline, "the container was never started"
        time.sleep(0.2)
    process.send_signal(signum)
    assert process.wait(timeout=30) == 128 + signum
    calls = fake_docker.read_text().splitlines()
    container = next(call for call in calls if call.startswith("run ")).split("--name ")[1].split()[0]
    assert f"rm -f {container}" in calls
