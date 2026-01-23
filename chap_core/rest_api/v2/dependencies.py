from functools import lru_cache

from chap_core.rest_api.services.orchestrator import Orchestrator
from chap_core.util import load_redis


@lru_cache
def get_redis():
    return load_redis(db=3)


def get_orchestrator():
    return Orchestrator(redis_client=get_redis())
