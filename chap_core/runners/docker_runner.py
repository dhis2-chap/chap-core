import logging
from pathlib import Path

from chap_core.runners.command_line_runner import CommandLineTrainPredictRunner

from ..docker_helper_functions import (
    run_command_through_docker_container,
)
from .runner import Runner

logger = logging.getLogger(__name__)


class DockerRunner(Runner):
    """Runs through a docker image specified by name (e.g. on dockerhub), not a Dockerfile"""

    def __init__(
        self, docker_name: str, working_dir: str | Path, model_configuration_filename: str | None = None, dry_run=False
    ):
        super().__init__(dry_run=dry_run)
        self._docker_name = docker_name
        self._working_dir = working_dir
        self._model_configuration_filename = model_configuration_filename

    def _execute(self, command, working_dir, env=None):
        if self._dry_run:
            print(f"[dry-run] docker run {self._docker_name} cd {working_dir} && {command}")
            return ""
        return run_command_through_docker_container(self._docker_name, str(working_dir), command)

    def run_command(self, command):
        logger.debug(f"Running command {command} in docker container {self._docker_name} in {self._working_dir}")
        return self._execute(command, self._working_dir)

    def teardown(self):
        # remove the docker image
        # client = docker.from_env()
        # client.images.remove(self._docker_name, force=True)
        pass


class DockerTrainPredictRunner(CommandLineTrainPredictRunner):
    """This is basically a CommandLineTrainPredictRunner, but with a DockerRunner
    instead of a CommandLineRunner as runner"""

    def __init__(
        self,
        runner: DockerRunner,
        train_command: str,
        predict_command: str,
        model_configuration_filename: str | None = None,
    ):
        # assert False, (predict_command, model_configuration_filename)

        super().__init__(runner, train_command, predict_command, model_configuration_filename)

    def teardown(self):
        self._runner.teardown()
