import abc


class Runner:
    """
    An interface for Runners. A runner is able to run "something", e.g. a command on the command line
    through Docker."""

    def __init__(self, dry_run=False):
        self._dry_run = dry_run

    def run_command(self, command): ...

    def _execute(self, command, working_dir, env=None):
        if self._dry_run:
            print(f"[dry-run] cd {working_dir} && {command}")
            return ""
        from chap_core.runners.command_line_runner import run_command

        return run_command(command, working_dir, env=env)

    def store_file(self, file_path: str | None = None) -> None:
        ...
        # not used for anything now

    def teardown(self):
        """To be called after the runner is done with train and predict. This is to clean up the runner, e.g.
        to remove docker images, etc"""


class TrainPredictRunner(abc.ABC):
    """
    Specific wrapper for runners that only run train/predict commands
    """

    @abc.abstractmethod
    def train(self, train_data: str, model_file_name: str, polygons_file_name: str | None): ...

    @abc.abstractmethod
    def predict(
        self,
        model_file_name: str,
        historic_data: str,
        future_data: str,
        output_file: str,
        polygons_file_name: str | None,
    ): ...

    def report(
        self,
        model_file_name: str,
        historic_data: str,
        output_file: str,
        polygons_file_name: str | None = None,
    ):
        raise NotImplementedError("This runner does not support report generation")

    def teardown(self):  # noqa: B027
        ...
