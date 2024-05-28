from pathlib import Path

from ..docker_helper_functions import create_docker_image, run_command_through_docker_container
from .runner import Runner


class DockerRunner(Runner):
    def __init__(self, docker_file_path: str, working_dir: str | Path):
        self._docker_file_path = docker_file_path
        self._docker_name = None
        self._working_dir = working_dir
        self._is_setup = False

    def setup(self):
        if self._is_setup:
            return
        self._docker_name = create_docker_image(self._docker_file_path, working_dir=self._working_dir)
        self._is_setup = True

    def run_command(self, command):
        self.setup()
        return run_command_through_docker_container(self._docker_name, self._working_dir, command)
