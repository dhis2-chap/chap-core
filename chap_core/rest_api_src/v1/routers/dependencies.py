import os
from functools import lru_cache

from sqlmodel import Session

from chap_core.database.database import engine
from chap_core.google_earth_engine.gee_era5 import Era5LandGoogleEarthEngine
from chap_core.rest_api_src.worker_functions import WorkerConfig


# TODO: make dependency injection in celery worker
def get_session():
    with Session(engine) as session:
        yield session


def get_session_wrapper():
    ...


def get_gee():
    '''
    Returns an instance of Era5LandGoogleEarthEngine
    '''
    return Era5LandGoogleEarthEngine()


@lru_cache
def get_settings():
    return WorkerConfig()


def get_database_url():
    return os.getenv("CHAP_DATABASE_URL")
