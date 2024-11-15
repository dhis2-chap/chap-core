import logging
import os
from pathlib import Path

#from chap_core.cli import logger
logger = logging.getLogger()

def initialize_logging(debug: bool=False, log_file: str=None):
    if debug:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("Debug mode enabled")

        # check if environment variable CHAP_LOG_FILE is set, use that as handler
    if os.getenv("CHAP_LOG_FILE"):
        #logger.addHandler(logging.FileHandler(os.getenv("CHAP_LOG_FILE")))
        #logger.info(f"Logging to {os.getenv('CHAP_LOG_FILE')}")
        log_file = os.getenv("CHAP_LOG_FILE")
        print("Overwriting log file to specified env variable ", log_file)

    if log_file is not None:
        # create file if not exist
        if not Path(log_file).exists():
            print(f"Creating log file at {log_file}")
            Path(log_file).touch()

        logger.addHandler(logging.FileHandler(log_file))
        logger.info(f"Logging to {log_file}")
