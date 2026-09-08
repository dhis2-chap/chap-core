import logging
import subprocess
from pathlib import Path

from chap_core.exceptions import CommandLineException, ModelConfigurationException
from chap_core.runners.runner import Runner, TrainPredictRunner

logger = logging.getLogger(__name__)


class CommandLineRunner(Runner):
    def __init__(self, working_dir: str | Path, dry_run=False):
        super().__init__(dry_run=dry_run)
        self._working_dir = working_dir

    def run_command(self, command):
        return self._execute(command, self._working_dir)

    def store_file(self, file_path: str | None = None) -> None:
        pass


def run_command(command: str, working_directory=Path("."), env: dict | None = None):
    """Runs a unix command using subprocess.

    Parameters
    ----------
    command : str
        The command to run
    working_directory : Path
        The directory to run the command in
    env : dict, optional
        Environment variables to use. If None, uses the current environment.
    """
    logging.debug(f"Running command: {command}")
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=working_directory, shell=True, env=env
    )
    stdout, stderr = process.communicate()
    # Model output is not guaranteed to be valid UTF-8 (locale-dependent R
    # warnings, for instance); a failed model must still produce a readable
    # error message rather than a UnicodeDecodeError.
    output = stdout.decode(errors="replace") + "\n" + stderr.decode(errors="replace")
    return_code = process.returncode

    if return_code != 0:
        message = (
            f"Command '{command}' failed with return code {return_code}, "
            f"Full output from command below: \n ----- \n{output} \n--------"
        )
        logger.error(message)
        raise CommandLineException(message)

    return output


class CommandLineTrainPredictRunner(TrainPredictRunner):
    def __init__(
        self,
        runner: Runner,
        train_command: str,
        predict_command: str,
        model_configuration_filename: str | None = None,
        report_command: str | None = None,
    ):
        self._runner = runner
        self._train_command = train_command
        self._predict_command = predict_command
        self._model_configuration_filename = model_configuration_filename
        self._report_command = report_command

    def _format_command(self, command, keys):
        try:
            return command.format(**keys)
        except KeyError as e:
            raise ModelConfigurationException(
                f"Was not able to format command {command}. Does the command contain wrong keys or keys that there is not data for in the dataset?"
            ) from e

    def _handle_polygons(self, command, keys, polygons_file_name=None):
        # adds polygons to keys if polygons exist. Does some checking with compatibility with command
        if polygons_file_name is not None:
            if "{polygons}" not in command:
                logger.warning(
                    f"Dataset has polygons, but command {command} does not ask for polygons. Will not insert polygons into command."
                )
            else:
                keys["polygons"] = polygons_file_name
        return keys

    def _handle_config(self, command, keys):
        if "{model_config}" not in command:
            return keys
        keys["model_config"] = self._model_configuration_filename
        return keys

    def train(self, train_file_name, model_file_name, polygons_file_name=None):
        keys = {"train_data": train_file_name, "model": model_file_name}
        keys = self._handle_polygons(self._train_command, keys, polygons_file_name)
        keys = self._handle_config(self._train_command, keys)
        command = self._format_command(self._train_command, keys)
        logger.debug(f"Running command {command}")
        return self._runner.run_command(command)

    def predict(self, model_file_name, historic_data, future_data, output_file, polygons_file_name=None):
        keys = {
            "historic_data": historic_data,
            "future_data": future_data,
            "model": model_file_name,
            "out_file": output_file,
        }
        keys = self._handle_polygons(self._predict_command, keys, polygons_file_name)
        keys = self._handle_config(self._predict_command, keys)
        command = self._format_command(self._predict_command, keys)
        return self._runner.run_command(command)

    def report(self, model_file_name, historic_data, output_file, polygons_file_name=None):
        if self._report_command is None:
            raise NotImplementedError("This runner does not support report generation")
        keys = {
            "model": model_file_name,
            "historic_data": historic_data,
            "out_file": output_file,
        }
        keys = self._handle_polygons(self._report_command, keys, polygons_file_name)
        keys = self._handle_config(self._report_command, keys)
        command = self._format_command(self._report_command, keys)
        logger.debug(f"Running command {command}")
        return self._runner.run_command(command)
