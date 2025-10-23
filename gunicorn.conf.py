# gunicorn.conf.py
import logging
import sys
import traceback
from chap_core.database.database import create_db_and_tables

# Configure logging to stdout/stderr for Docker
logconfig_dict = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'generic': {
            'format': '%(levelname)s [%(name)s] %(message)s',
            'class': 'logging.Formatter',
        },
        'detailed': {
            'format': '%(asctime)s [%(process)d] [%(levelname)s] [%(name)s] %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S',
            'class': 'logging.Formatter',
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'detailed',
            'stream': sys.stdout,
        },
        'error_console': {
            'class': 'logging.StreamHandler',
            'formatter': 'detailed',
            'stream': sys.stderr,
        },
    },
    'root': {
        'level': 'INFO',
        'handlers': ['console'],
    },
    'loggers': {
        'gunicorn.error': {
            'level': 'INFO',
            'handlers': ['error_console'],
            'propagate': False,
        },
        'gunicorn.access': {
            'level': 'INFO',
            'handlers': ['console'],
            'propagate': False,
        },
    },
}

# Set log level
loglevel = 'info'

# Capture stdout to ensure print statements appear in logs
capture_output = True

log = logging.getLogger("gunicorn.error")


def on_starting(server):
    log.info("Database migration and initialization starting...")
    create_db_and_tables()
    log.info("Database migration and initialization complete.")


def worker_abort(worker):
    """Called when a worker is aborted (e.g., due to timeout or exception)"""
    worker.log.error(f"Worker {worker.pid} aborted")
    worker.log.error(traceback.format_exc())


def worker_int(worker):
    """Called when a worker receives an INT or QUIT signal"""
    worker.log.info(f"Worker {worker.pid} received INT/QUIT signal")
