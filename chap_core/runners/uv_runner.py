import logging
import os
from pathlib import Path

from chap_core.runners.command_line_runner import CommandLineTrainPredictRunner, run_command
from .runner import Runner

logger = logging.getLogger(__name__)


class UvRunner(Runner):
    """Runs commands through uv in a pyproject.toml-managed environment"""

    def __init__(self, working_dir: str | Path):
        self._working_dir = Path(working_dir)

    def run_command(self, command):
        uv_command = f"uv run {command}"
        logger.debug(f"Running command {uv_command} in {self._working_dir}")
        # Override UV_PROJECT_ENVIRONMENT to use a model-specific venv in the working directory.
        # This prevents UV from trying to install packages into the main application's venv.
        env = os.environ.copy()
        env["UV_PROJECT_ENVIRONMENT"] = str(self._working_dir / ".venv")
        return run_command(uv_command, self._working_dir, env=env)

    def store_file(self):
        pass

    def teardown(self):
        pass


class UvTrainPredictRunner(CommandLineTrainPredictRunner):
    """CommandLineTrainPredictRunner that uses UvRunner to execute commands via uv"""

    def __init__(
        self,
        runner: UvRunner,
        train_command: str,
        predict_command: str,
        model_configuration_filename: str | None = None,
    ):
        super().__init__(runner, train_command, predict_command, model_configuration_filename)

    def teardown(self):
        self._runner.teardown()
