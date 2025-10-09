# gunicorn.conf.py
import logging
from chap_core.database.database import create_db_and_tables

log = logging.getLogger("gunicorn.error")


def on_starting(server):
    log.info("Database migration and initialization starting...")
    create_db_and_tables()
    log.info("Database migration and initialization complete.")
