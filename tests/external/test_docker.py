import docker.errors
import pytest
from chap_core.docker_helper_functions import (
    create_docker_image,
    run_command_through_docker_container,
)
from chap_core.exceptions import DockerUnavailableError
from chap_core.util import docker_available


@pytest.mark.skipif(not docker_available(), reason="Docker not available")
def test_run_docker_basic():
    result = run_command_through_docker_container("ubuntu", "./", "echo 'hi'")

    with pytest.raises(docker.errors.APIError):
        result = run_command_through_docker_container("ubuntu", "./", "command_not_existing", remove_after_run=True)


def _raise_docker_unavailable(*_args, **_kwargs):
    raise docker.errors.DockerException(
        "Error while fetching server API version: ('Connection aborted.', "
        "FileNotFoundError(2, 'No such file or directory'))"
    )


def test_docker_unavailable_in_server(monkeypatch):
    monkeypatch.setenv("IS_IN_DOCKER", "1")
    monkeypatch.setattr("chap_core.docker_helper_functions.docker.from_env", _raise_docker_unavailable)
    with pytest.raises(DockerUnavailableError) as excinfo:
        run_command_through_docker_container("rwanda_malaria_bym", "./", "echo hi")
    msg = str(excinfo.value)
    assert "rwanda_malaria_bym" in msg
    assert "chapkit" in msg
    assert "docker_env" in msg


def test_docker_unavailable_locally(monkeypatch):
    monkeypatch.delenv("IS_IN_DOCKER", raising=False)
    monkeypatch.setattr("chap_core.docker_helper_functions.docker.from_env", _raise_docker_unavailable)
    with pytest.raises(DockerUnavailableError) as excinfo:
        run_command_through_docker_container("some_model", "./", "echo hi")
    msg = str(excinfo.value)
    assert "Docker Desktop" in msg or "systemctl start docker" in msg
