"""Read-only reverse proxy from CHAP Core to registered chapkit service instances.

Chapkit model services register with the v2 orchestrator and advertise a base URL that
is only reachable from within CHAP Core's network. This router exposes a catch-all route
that forwards read-only (GET/HEAD) requests under ``/v2/services/{service_id}/run/{path}``
to the registered service, streaming the response back so binary artifact downloads work
without buffering. It is the foundation for read-only features built on top of chapkit
(artifact browser, config inspection, job viewer).

The proxy is intentionally limited to safe methods: mutating verbs are not exposed because
the route is unauthenticated. Adding mutations later requires an explicit authorization
boundary first.
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

# Read-only only. FastAPI does not auto-add HEAD, so it is listed explicitly.
PROXY_METHODS = ["GET", "HEAD"]


@router.api_route(
    "/{service_id}/run/{path:path}",
    methods=PROXY_METHODS,
    summary="Proxy a read-only request to a registered chapkit service",
)
async def proxy_to_service(
    service_id: str,
    path: str,
    request: Request,
    orchestrator: Orchestrator = Depends(get_orchestrator),
    client: httpx.AsyncClient = Depends(get_http_client),
) -> StreamingResponse:
    """Forward a read-only request to a registered chapkit service and stream the response back.

    The path below ``run/`` is forwarded verbatim, so callers use the service's real
    paths, e.g. ``/v2/services/{id}/run/api/v1/artifacts``. Returns 404 if the service id
    is unknown, and 502/504 if the service is unreachable or times out.
    """
    try:
        service = orchestrator.get(service_id)
    except ServiceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e

    target_url = service.url.rstrip("/") + "/" + path

    # Forward request headers minus hop-by-hop and host (httpx re-derives host).
    request_headers = [
        (k, v) for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP_HEADERS and k.lower() != "host"
    ]

    upstream_request = client.build_request(
        request.method,
        target_url,
        params=request.query_params,
        headers=request_headers,
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

    response = StreamingResponse(
        upstream.aiter_raw(),
        status_code=upstream.status_code,
        background=BackgroundTask(upstream.aclose),
    )
    # Use multi_items() and assign raw_headers so repeated headers (e.g. Set-Cookie) are
    # preserved as distinct entries rather than collapsed into one comma-joined value.
    response.raw_headers = [
        (k.encode("latin-1"), v.encode("latin-1"))
        for k, v in upstream.headers.multi_items()
        if k.lower() not in HOP_BY_HOP_HEADERS
    ]
    return response
