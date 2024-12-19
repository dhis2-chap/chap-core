import logging
import os
from pathlib import Path

# get root logger
logger = logging.getLogger()
_global_log_file = None


def initialize_logging(debug: bool=False, log_file: str=None):
    if debug:
        logger.setLevel(level=logging.DEBUG)
        logger.debug("Debug mode enabled")
    else:
        logger.setLevel(level=logging.INFO)
        logger.info("Level set to INFO")

    # check if environment variable CHAP_LOG_FILE is set, use that as handler
    if os.getenv("CHAP_LOG_FILE") and log_file is None:
        log_file = os.getenv("CHAP_LOG_FILE")
        print("Overwriting log file to specified env variable ", log_file)

    if log_file is not None:
        # create file if not exist
        if not Path(log_file).exists():
            print(f"Creating log file at {log_file}")
            logging.info(f"Creating log file at {log_file}")
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            Path(log_file).touch()
            os.chmod(log_file, 0o664)

        logger.addHandler(logging.FileHandler(log_file))
        logger.info(f"Logging to {log_file}")
        print("Willl log to ", log_file)
        global _global_log_file
        _global_log_file = log_file


def get_log_file_path():
    return _global_log_file


def get_logs():
    if _global_log_file is not None:
        with open(_global_log_file, 'r') as f:
            return f.read()
