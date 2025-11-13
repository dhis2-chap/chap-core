import logging
import os
from pathlib import Path

# get root logger
logger = logging.getLogger()


def is_debug_mode() -> bool:
    """Check if CHAP_DEBUG environment variable is set to enable debug mode.

    Returns:
        bool: True if CHAP_DEBUG is set to "true", "1", or "yes" (case-insensitive).
    """
    return os.getenv("CHAP_DEBUG", "false").lower() in ("true", "1", "yes")


def initialize_logging(debug: bool = None, log_file: str = None):
    """Initialize logging configuration.

    Args:
        debug: If True, set log level to DEBUG. If None, auto-detect from CHAP_DEBUG environment variable.
        log_file: Optional log file path (currently not used).
    """
    # Auto-detect from environment if not explicitly provided
    if debug is None:
        debug = is_debug_mode()

    if debug:
        logger.setLevel(level=logging.DEBUG)
        logger.debug("Debug mode enabled")
    else:
        logger.setLevel(level=logging.INFO)
        logger.info("Level set to INFO")

    Path("logs").mkdir(parents=True, exist_ok=True)  # keep this in as it might be required for celery tasks

    # check if environment variable CHAP_LOG_FILE is set, use that as handler
    # if os.getenv("CHAP_LOG_FILE") and log_file is None:
    #     log_file = os.getenv("CHAP_LOG_FILE")
    #     print("Overwriting log file to specified env variable ", log_file)

    # if log_file is not None:
    #     # create file if not exist
    #     if not Path(log_file).exists():
    #         print(f"Creating log file at {log_file}")
    #         logging.info(f"Creating log file at {log_file}")
    #         Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    #         Path(log_file).touch()
    #         os.chmod(log_file, 0o664)

    #     logger.addHandler(logging.FileHandler(log_file))
    #     logger.info(f"Logging to {log_file}")
    #     print("Willl log to ", log_file)
    #     global _global_log_file
    #     _global_log_file = log_file


# def get_log_file_path():
#     return _global_log_file


# def get_logs():
#     if _global_log_file is not None:
#         with open(_global_log_file, "r") as f:
#             return f.read()
