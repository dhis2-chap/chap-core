"""Blocking route handlers run in the threadpool, so they never hold up other requests (CLIM-1810)."""

import inspect

import anyio
from anyio import to_thread
from fastapi.routing import APIRoute

from chap_core.database.database import DB_POOL_CONNECTIONS
from chap_core.rest_api.app import app, lifespan

# Async handlers that do no blocking work, or that await their blocking work in a thread.
ASYNC_HANDLERS = {"health", "readiness", "deep_probe", "system_info", "favicon", "proxy_to_service"}


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
