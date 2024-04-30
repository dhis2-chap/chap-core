import pytest
from climate_health.docker_helper_functions import create_docker_image, run_command_through_docker_container
from climate_health.util import docker_available


@pytest.mark.skipif(not docker_available(), reason="Docker not available")
def test_create_inla_image(models_path):
    docker_directory = models_path / 'docker_r_base'
    name = create_docker_image(docker_directory)
    assert name == "docker_r_base"

    # test that INLA can be loaded
    testcommand = "R -e 'print(\"test1\"); library(INLA); print(\"test2\")'"
    result = run_command_through_docker_container(name, "./", testcommand)
    assert "This is INLA" in result
