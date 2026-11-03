import os
import signal
import subprocess
import sys
import time

import httpx
import pytest

from chap_core.exceptions import ChapkitServicePortInUseError
from chap_core.models.chapkit_service_manager import DockerChapkitServiceManager, find_available_port
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


def test_each_launch_gets_its_own_container_name(fake_docker):
    port = find_available_port()
    for _ in range(2):
        with DockerChapkitServiceManager(IMAGE, port=port, startup_timeout=15):
            pass
    runs = [call for call in fake_docker.read_text().splitlines() if call.startswith("run ")]
    names = {run.split("--name ")[1].split()[0] for run in runs}
    assert len(runs) == 2
    assert len(names) == 2


def test_a_container_whose_port_is_taken_is_removed(fake_docker, monkeypatch):
    monkeypatch.setenv("FAKE_DOCKER_PORT_TAKEN", "1")
    with pytest.raises(ChapkitServicePortInUseError):
        with DockerChapkitServiceManager(IMAGE, port=find_available_port(), startup_timeout=15):
            pass
    calls = fake_docker.read_text().splitlines()
    run = next(call for call in calls if call.startswith("run "))
    container = run.split("--name ")[1].split()[0]
    assert calls[-1] == f"rm -f {container}"


def test_a_service_from_another_run_on_the_same_port_is_not_used(fake_docker, other_healthy_service, monkeypatch):
    monkeypatch.setenv("FAKE_DOCKER_PORT_TAKEN", "1")
    monkeypatch.setenv("FAKE_DOCKER_STARTUP_SECONDS", "2")
    with pytest.raises(ChapkitServicePortInUseError):
        with DockerChapkitServiceManager(IMAGE, port=other_healthy_service, startup_timeout=15):
            pass


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


def test_a_command_started_with_sighup_ignored_survives_sighup(fake_docker, monkeypatch):
    monkeypatch.setenv("FAKE_DOCKER_STARTUP_SECONDS", "60")
    command = [sys.executable, "-m", "chap_core.cli", "model", "schema", f"docker://{IMAGE}"]

    def ignore_sighup():  # what nohup does
        signal.signal(signal.SIGHUP, signal.SIG_IGN)

    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=ignore_sighup)
    deadline = time.monotonic() + 60
    while not (fake_docker.exists() and "run " in fake_docker.read_text()):
        assert time.monotonic() < deadline, "the container was never started"
        time.sleep(0.2)
    process.send_signal(signal.SIGHUP)
    time.sleep(2)
    assert process.poll() is None
    process.send_signal(signal.SIGTERM)
    assert process.wait(timeout=30) == 128 + signal.SIGTERM
