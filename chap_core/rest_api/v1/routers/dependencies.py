import os
from functools import lru_cache

from sqlmodel import Session

from chap_core.database.database import engine
from chap_core.rest_api.services.model_service import ModelService
from chap_core.rest_api.services.orchestrator import Orchestrator
from chap_core.rest_api.worker_functions import WorkerConfig
from chap_core.util import load_redis


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


@lru_cache
def get_redis():
    """Get Redis client for service registration."""
    return load_redis(db=3)


def get_orchestrator() -> Orchestrator:
    """
    Dependency that provides Orchestrator for accessing registered services.
    """
    return Orchestrator(redis_client=get_redis())


def get_model_service(
    session: Session = None,
    orchestrator: Orchestrator = None,
) -> ModelService:
    """
    Dependency that provides ModelService for retrieving all models.

    Combines database session for static models with orchestrator
    for dynamic registered services.
    """
    # This function is used as a dependency, but FastAPI's Depends()
    # doesn't work here directly, so we need to import and use it
    # in the route that needs it
    return ModelService(session=session, orchestrator=orchestrator)
