import logging
import subprocess
import sys
from pathlib import Path
from chap_core.runners.runner import Runner


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
        print(command)
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, cwd=working_directory, shell=True
        )
        output = []
        for c in iter(lambda: process.stdout.read(1), b""):
            sys.stdout.buffer.write(c)
            output.append(c.decode("ascii"))
        print("finished")
        streamdata = process.communicate()[0]  # finnish before getting return code
        return_code = process.returncode
        assert (
            return_code == 0
        ), f"Command '{command}' failed with return code {return_code}, ({''.join(streamdata)}, {''.join(output)}"
        # output = subprocess.check_output(' '.join(command), cwd=working_directory, shell=True)
        # logging.info(output)
    except subprocess.CalledProcessError as e:
        error = e.output.decode()
        logging.info(error)
        raise e
