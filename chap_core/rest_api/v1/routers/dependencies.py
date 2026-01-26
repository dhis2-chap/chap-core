import os
from functools import lru_cache

from sqlmodel import Session

from chap_core.database.database import engine
from chap_core.rest_api.worker_functions import WorkerConfig


# TODO: make dependency injection in celery worker
def get_session():
    with Session(engine) as session:
        yield session


def get_session_wrapper(): ...


@lru_cache
def get_settings():
    return WorkerConfig()


def get_database_url():
    return os.getenv("CHAP_DATABASE_URL")
