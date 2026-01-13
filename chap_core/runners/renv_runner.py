import logging
from pathlib import Path

from chap_core.exceptions import CommandLineException
from chap_core.runners.command_line_runner import CommandLineTrainPredictRunner, run_command
from .runner import Runner

logger = logging.getLogger(__name__)


class RenvRunner(Runner):
    """Runs R commands in an renv-managed environment.

    renv auto-activates via .Rprofile when Rscript is started in the project directory.
    This runner ensures dependencies are restored before command execution.
    """

    def __init__(self, working_dir: str | Path, auto_restore: bool = True):
        self._working_dir = Path(working_dir)
        self._auto_restore = auto_restore
        self._restored = False

    def _ensure_restored(self):
        """Run renv::restore() if not already done and auto_restore is enabled."""
        if self._restored or not self._auto_restore:
            return

        renv_lock = self._working_dir / "renv.lock"
        if not renv_lock.exists():
            raise CommandLineException(
                f"renv.lock not found in {self._working_dir}. "
                "Ensure the R project has been initialized with renv::init()."
            )

        logger.info(f"Restoring renv environment in {self._working_dir}")
        restore_command = 'Rscript -e "renv::restore(prompt = FALSE)"'
        run_command(restore_command, self._working_dir)
        self._restored = True

    def run_command(self, command):
        self._ensure_restored()
        logger.debug(f"Running command {command} in {self._working_dir}")
        return run_command(command, self._working_dir)

    def store_file(self):
        pass

    def teardown(self):
        pass


class RenvTrainPredictRunner(CommandLineTrainPredictRunner):
    """CommandLineTrainPredictRunner that uses RenvRunner to execute R commands via renv."""

    def __init__(
        self,
        runner: RenvRunner,
        train_command: str,
        predict_command: str,
        model_configuration_filename: str | None = None,
    ):
        super().__init__(runner, train_command, predict_command, model_configuration_filename)

    def teardown(self):
        self._runner.teardown()
