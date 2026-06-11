"""Transparent reverse proxy from CHAP Core to registered chapkit service instances.

Chapkit model services register with the v2 orchestrator and advertise a base URL that
is only reachable from within CHAP Core's network. This router exposes a catch-all route
that forwards any request under ``/v2/services/{service_id}/run/{path}`` to the registered
service, streaming both request and response so binary artifact downloads work without
buffering. It is the foundation for building higher-level features (artifact browser,
config UI, job viewer) on top of chapkit.
"""

import logging

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from chap_core.rest_api.services.orchestrator import Orchestrator, ServiceNotFoundError
from chap_core.rest_api.v2.dependencies import get_http_client, get_orchestrator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/services", tags=["Services"])

# Hop-by-hop headers must not be forwarded by a proxy (RFC 7230 section 6.1).
HOP_BY_HOP_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)

# HEAD is intentionally omitted: Starlette auto-adds it alongside GET. Listing it
# explicitly would register the route twice and produce a duplicate OpenAPI operation id.
PROXY_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE"]


def _filter_headers(headers, drop: set[str]) -> list[tuple[str, str]]:
    """Drop hop-by-hop and explicitly excluded headers (case-insensitive)."""
    return [(k, v) for k, v in headers.items() if k.lower() not in HOP_BY_HOP_HEADERS and k.lower() not in drop]


@router.api_route(
    "/{service_id}/run/{path:path}",
    methods=PROXY_METHODS,
    summary="Proxy a request to a registered chapkit service",
)
async def proxy_to_service(
    service_id: str,
    path: str,
    request: Request,
    orchestrator: Orchestrator = Depends(get_orchestrator),
    client: httpx.AsyncClient = Depends(get_http_client),
) -> StreamingResponse:
    """Forward an arbitrary request to a registered chapkit service and stream the response back.

    The path below ``run/`` is forwarded verbatim, so callers use the service's real
    paths, e.g. ``/v2/services/{id}/run/api/v1/artifacts``. Returns 404 if the service id
    is unknown, and 502/504 if the service is unreachable or times out.
    """
    try:
        service = orchestrator.get(service_id)
    except ServiceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e

    target_url = service.url.rstrip("/") + "/" + path

    # Let httpx re-derive host/content-length for the upstream request.
    request_headers = _filter_headers(request.headers, drop={"host", "content-length"})
    body = await request.body()

    upstream_request = client.build_request(
        request.method,
        target_url,
        params=request.query_params,
        headers=request_headers,
        content=body,
    )

    try:
        upstream = await client.send(upstream_request, stream=True)
    except httpx.TimeoutException as e:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"Service {service_id} did not respond in time",
        ) from e
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Service {service_id} is unreachable: {e}",
        ) from e

    response_headers = _filter_headers(upstream.headers, drop=set())
    return StreamingResponse(
        upstream.aiter_raw(),
        status_code=upstream.status_code,
        headers=dict(response_headers),
        background=BackgroundTask(upstream.aclose),
    )
