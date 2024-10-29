import docker.errors
import pytest
from chap_core.docker_helper_functions import (
    create_docker_image,
    run_command_through_docker_container,
)
from chap_core.util import docker_available


@pytest.mark.skipif(not docker_available(), reason="Docker not available")
@pytest.mark.slow
@pytest.mark.skip(reason="Not necessary anymore, this image is not being used directly")
def test_create_inla_image(models_path):
    docker_directory = models_path / "docker_r_base"
    name = create_docker_image(docker_directory)
    assert name == "docker_r_base"

    # test that INLA can be loaded
    testcommand = 'R -e \'print("test1"); library(INLA); print("test2")\''
    result = run_command_through_docker_container(name, "./", testcommand)
    assert "This is INLA" in result


@pytest.mark.skipif(not docker_available(), reason="Docker not available")
def test_run_docker_basic(models_path):
    result = run_command_through_docker_container("ubuntu", "./", "echo 'hi'")

    with pytest.raises(docker.errors.APIError):
        result = run_command_through_docker_container(
            "ubuntu", "./", "command_not_existing", remove_after_run=True
        )
