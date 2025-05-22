import logging
import subprocess
from pathlib import Path
from chap_core.exceptions import CommandLineException, ModelConfigurationException
from chap_core.runners.runner import Runner, TrainPredictRunner

logger = logging.getLogger(__name__)


class CommandLineRunner(Runner):
    def __init__(self, working_dir: str | Path):
        self._working_dir = working_dir

    def run_command(self, command):
        return run_command(command, self._working_dir)

    def store_file(self):
        pass


def run_command(command: str, working_directory=Path(".")):
    """Runs a unix command using subprocess"""
    logging.info(f"Running command: {command}")
    # command = command.split()

    try:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=working_directory, shell=True
        )
        stdout, stderr = process.communicate()
        output = stdout.decode() + "\n" + stderr.decode()
        """
        output = []
        for c in iter(lambda: process.stdout.read(1), b""):
            sys.stdout.buffer.write(c)
            output.append(c.decode("utf-8"))
        for c in iter(lambda: process.stderr.read(1), b""):
            sys.stderr.buffer.write(c)
            output.append(c.decode("utf-8"))
        output = ''.join(output)
        """

        streamdata = process.communicate()[0]  # finnish before getting return code
        return_code = process.returncode

        if return_code != 0:
            logger.error(
                f"Command '{command}' failed with return code {return_code}, ({''.join(map(str, streamdata))}, {output}"
            )
            raise CommandLineException(
                f"Command '{command}' failed with return code {return_code}, Full output from command below: \n ----- \n({''.join(map(str, streamdata))}, {output} \n--------"
            )
        # output = subprocess.check_output(' '.join(command), cwd=working_directory, shell=True)
        # logging.info(output)
    except subprocess.CalledProcessError as e:
        error = e.output.decode()
        logger.info(error)
        raise e

    return output


class CommandLineTrainPredictRunner(TrainPredictRunner):
    def __init__(
        self,
        runner: CommandLineRunner,
        train_command: str,
        predict_command: str,
        model_configuration_filename: str | None = None,
    ):
        self._runner = runner
        self._train_command = train_command
        self._predict_command = predict_command
        self._model_configuration_filename = model_configuration_filename

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
        logger.info(f"Running command {command}")
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
