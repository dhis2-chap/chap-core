from pathlib import Path
import docker
from ..docker_helper_functions import (
    create_docker_image,
    run_command_through_docker_container,
)
from .runner import Runner
import logging

logger = logging.getLogger(__name__)


class DockerImageRunner(Runner):
    """A runner based on a docker image (Dockerfile)"""

    def __init__(self, docker_file_path: str, working_dir: str | Path):
        self._docker_file_path = Path(working_dir) / docker_file_path
        self._docker_name = None
        self._working_dir = working_dir
        self._is_setup = False

    def setup(self):
        if self._is_setup:
            return
        self._docker_name = create_docker_image(self._docker_file_path)
        self._is_setup = True

    def run_command(self, command):
        self.setup()
        return run_command_through_docker_container(self._docker_name, self._working_dir, command)


class DockerRunner(Runner):
    """Runs through a docker image specified by name (e.g. on dockerhub), not a Dockerfile"""

    def __init__(self, docker_name: str, working_dir: str | Path):
        self._docker_name = docker_name
        self._working_dir = working_dir

    def run_command(self, command):
        logger.info(f"Running command {command} in docker container {self._docker_name} in {self._working_dir}")
        return run_command_through_docker_container(self._docker_name, self._working_dir, command)

    def teardown(self):
        # remove the docker image
        client = docker.from_env()
        client.images.remove(self._docker_name, force=True)

