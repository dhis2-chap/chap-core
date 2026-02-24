import logging
from pathlib import Path

from chap_core.runners.command_line_runner import CommandLineTrainPredictRunner, run_command

from .runner import Runner

logger = logging.getLogger(__name__)


class CondaRunner(Runner):
    """Runs commands inside a conda environment created from an environment YAML file."""

    def __init__(self, working_dir: str | Path, conda_env_file: str):
        self._working_dir = Path(working_dir)
        self._conda_env_file = conda_env_file
        self._env_path = self._working_dir / ".conda_env"
        self._env_created = False

    def _ensure_environment(self):
        if self._env_created:
            return

        env_file = self._working_dir / self._conda_env_file
        if not env_file.exists():
            raise FileNotFoundError(f"Conda environment file {self._conda_env_file} not found in {self._working_dir}")

        if self._env_path.exists():
            logger.info(f"Updating conda environment at {self._env_path}")
            cmd = f"conda env update -f {self._conda_env_file} -p {self._env_path}"
        else:
            logger.info(f"Creating conda environment at {self._env_path}")
            cmd = f"conda env create -f {self._conda_env_file} -p {self._env_path}"

        run_command(cmd, self._working_dir)
        self._env_created = True

    def run_command(self, command):
        self._ensure_environment()
        conda_command = f"conda run --no-capture-output -p {self._env_path} {command}"
        logger.debug(f"Running command {conda_command} in {self._working_dir}")
        return run_command(conda_command, self._working_dir)

    def store_file(self, file_path: str | None = None) -> None:
        pass

    def teardown(self):
        pass


class CondaTrainPredictRunner(CommandLineTrainPredictRunner):
    """CommandLineTrainPredictRunner that uses CondaRunner to execute commands via conda."""

    def __init__(
        self,
        runner: CondaRunner,
        train_command: str,
        predict_command: str,
        model_configuration_filename: str | None = None,
    ):
        super().__init__(runner, train_command, predict_command, model_configuration_filename)

    def teardown(self):
        self._runner.teardown()
