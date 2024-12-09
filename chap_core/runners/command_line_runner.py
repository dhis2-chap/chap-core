import logging
import subprocess
from pathlib import Path
from chap_core.exceptions import CommandLineException
from chap_core.runners.runner import Runner

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
        process = subprocess.Popen(command, stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE, 
                                   cwd=working_directory, shell=True)
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
            logger.error(f"Command '{command}' failed with return code {return_code}, ({''.join(map(str, streamdata))}, {output}")
            raise CommandLineException(f"Command '{command}' failed with return code {return_code}, Full output from command below: \n ----- \n({''.join(map(str, streamdata))}, {output} \n--------")
        # output = subprocess.check_output(' '.join(command), cwd=working_directory, shell=True)
        # logging.info(output)
    except subprocess.CalledProcessError as e:
        error = e.output.decode()
        logger.info(error)
        raise e

    return output