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

The path below ``run/`` is forwarded verbatim from the raw request path so reserved
characters in artifact ids and filenames survive, and upstream redirects are followed so a
relative ``Location`` resolves against the service rather than CHAP Core's own origin.
"""

import logging
from collections.abc import Iterable

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


def _connection_named_headers(header_pairs: Iterable[tuple[str, str]]) -> set[str]:
    """Header names nominated as hop-by-hop by a ``Connection`` header (RFC 7230 section 6.1)."""
    drop: set[str] = set()
    for key, value in header_pairs:
        if key.lower() != "connection":
            continue
        for token in value.split(","):
            normalized = token.strip().lower()
            if normalized and normalized not in ("close", "keep-alive"):
                drop.add(normalized)
    return drop


def _forwardable(header_pairs: Iterable[tuple[str, str]], drop: set[str]) -> list[tuple[str, str]]:
    """Filter out hop-by-hop, Connection-nominated, and explicitly excluded headers."""
    pairs = list(header_pairs)
    excluded = HOP_BY_HOP_HEADERS | drop | _connection_named_headers(pairs)
    return [(k, v) for k, v in pairs if k.lower() not in excluded]


def _raw_subpath(request: Request, service_id: str) -> str:
    """The still-percent-encoded path below ``run/``.

    FastAPI decodes ``{path}`` (turning ``a%2Fb`` into ``a/b``), so we slice the original
    encoded path out of the ASGI ``raw_path`` to forward reserved characters verbatim.
    """
    raw_path = request.scope.get("raw_path")
    if raw_path is None:
        return str(request.path_params["path"])
    text: str = raw_path.decode("ascii")
    marker = f"/{service_id}/run/"
    index = text.find(marker)
    if index == -1:
        return str(request.path_params["path"])
    return text[index + len(marker) :]


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

    # Forward request headers minus hop-by-hop, Connection-nominated, and host (httpx re-derives host).
    request_headers = _forwardable(request.headers.items(), drop={"host"})
    query_string = request.scope.get("query_string") or b""

    try:
        target = httpx.URL(service.url.rstrip("/") + "/" + _raw_subpath(request, service_id))
        if query_string:
            target = target.copy_with(query=query_string)
        upstream_request = client.build_request(request.method, target, headers=request_headers)
        # follow_redirects so a relative upstream Location resolves against the service.
        upstream = await client.send(upstream_request, stream=True, follow_redirects=True)
    except httpx.InvalidURL as e:
        # A malformed registered service URL (e.g. a bad port) is a misconfigured upstream.
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Service {service_id} has an invalid URL: {e}",
        ) from e
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
    # multi_items() + raw_headers so repeated headers (e.g. Set-Cookie) are preserved as
    # distinct entries rather than collapsed into one comma-joined value.
    response_headers = _forwardable(upstream.headers.multi_items(), drop=set())
    response.raw_headers = [(k.encode("latin-1"), v.encode("latin-1")) for k, v in response_headers]
    return response
