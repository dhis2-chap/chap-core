import logging
import subprocess
import sys
from pathlib import Path
from climate_health.runners.runner import Runner


class CommandLineRunner(Runner):
    def __init__(self, working_dir: str | Path):
        self._working_dir = working_dir

    def run_command(self, command):
        return run_command(command, self._working_dir)

    def store_file(self):
        pass


def run_command(command: str, working_directory="./"):
    """Runs a unix command using subprocess"""
    logging.info(f"Running command: {command}")
    # command = command.split()

    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                   cwd=working_directory, shell=True)
        for c in iter(lambda: process.stdout.read(1), b""):
            sys.stdout.buffer.write(c)
        print('finished')
        # output = subprocess.check_output(' '.join(command), cwd=working_directory, shell=True)
        # logging.info(output)
    except subprocess.CalledProcessError as e:
        error = e.output.decode()
        logging.info(error)
        raise e
