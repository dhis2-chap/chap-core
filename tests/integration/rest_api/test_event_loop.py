"""Blocking route handlers run in the threadpool, so they never hold up other requests (CLIM-1810)."""

import inspect
import threading
import time

import anyio
from anyio import to_thread
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from chap_core.database.database import DB_POOL_CONNECTIONS
from chap_core.rest_api import common_routes
from chap_core.rest_api.app import app, lifespan

# Async handlers that do no blocking work, or that await real async I/O.
ASYNC_HANDLERS = {"health", "system_info", "favicon", "proxy_to_service"}


class SlowRedis:
    def ping(self):
        time.sleep(3)


def test_slow_handler_does_not_block_health_check(monkeypatch):
    monkeypatch.setattr(common_routes, "load_redis", SlowRedis)
    monkeypatch.setattr(common_routes.celery_app.control, "ping", lambda timeout: [{"worker": "pong"}])

    # One client keeps one event loop, so both requests share it as on a real server.
    with TestClient(app) as client:
        slow = threading.Thread(target=client.get, args=("/health/ready",))
        slow.start()
        time.sleep(0.5)
        response = client.get("/health")
        # On the event loop, the slow check would have finished before /health could answer.
        still_running = slow.is_alive()
        slow.join()

    assert response.status_code == 200
    assert still_running


def test_threadpool_matches_database_pool():
    async def threadpool_size():
        async with lifespan(app):
            return to_thread.current_default_thread_limiter().total_tokens

    assert anyio.run(threadpool_size) == DB_POOL_CONNECTIONS


def api_routes(routes):
    # Included routers appear as wrappers around the router they include.
    for route in routes:
        if isinstance(route, APIRoute):
            yield route
        elif hasattr(route, "original_router"):
            yield from api_routes(route.original_router.routes)


def test_only_listed_handlers_are_async():
    routes = list(api_routes(app.routes))
    assert len(routes) > 50
    async_handlers = {route.endpoint.__name__ for route in routes if inspect.iscoroutinefunction(route.endpoint)}
    assert async_handlers <= ASYNC_HANDLERS, (
        "Make these handlers plain def so FastAPI runs them in the threadpool, "
        f"or add them to ASYNC_HANDLERS if they only await: {sorted(async_handlers - ASYNC_HANDLERS)}"
    )
