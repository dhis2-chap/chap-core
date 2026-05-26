import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlmodel import Session

from chap_core.rest_api.services.orchestrator import Orchestrator, ServiceNotFoundError
from chap_core.rest_api.services.schemas import (
    PingResponse,
    RegistrationRequest,
    RegistrationResponse,
    ServiceDetail,
    ServiceListResponse,
)
from chap_core.rest_api.v1.routers.dependencies import get_session
from chap_core.rest_api.v2.dependencies import get_orchestrator, verify_service_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/services", tags=["Services"])


@router.post(
    "/$register",
    response_model=RegistrationResponse,
    summary="Onboard a CHAPKit model service",
)
def register_service(
    payload: RegistrationRequest,
    request: Request,
    orchestrator: Orchestrator = Depends(get_orchestrator),
    session: Session = Depends(get_session),
    _: str = Depends(verify_service_key),
) -> RegistrationResponse:
    """Announce a CHAPKit-hosted model service so CHAP Core can route work to it.

    The orchestrator records the service and returns the absolute ping URL the service
    must hit periodically to stay live. As a side effect, the service's templates and
    default configurations are eagerly pulled into the v1 CRUD tables, so backtests and
    predictions can target it without waiting for the next lazy sync. Requires the
    ``X-Service-Key`` header.
    """
    response = orchestrator.register(payload)
    response.ping_url = str(request.base_url).rstrip("/") + response.ping_url

    # Eagerly sync the chapkit service into the DB so that model templates
    # and configured models are immediately queryable via the v1 CRUD
    # endpoints — no need to wait for a lazy GET /v1/crud/model-templates.
    # Best-effort: a sync failure must not fail the registration itself.
    try:
        from chap_core.rest_api.v1.routers.crud import _sync_live_chapkit_services

        _sync_live_chapkit_services(session)
    except Exception:
        logger.warning("Eager chapkit DB sync after registration failed", exc_info=True)

    return response


@router.put(
    "/{service_id}/$ping",
    response_model=PingResponse,
    summary="Send a keepalive heartbeat for a service",
)
def ping_service(
    service_id: str,
    orchestrator: Orchestrator = Depends(get_orchestrator),
    _: str = Depends(verify_service_key),
) -> PingResponse:
    """Tell the orchestrator the service is still alive so it is not evicted from the registry.

    Called by the CHAPKit service itself on a timer — typically once a minute. The
    orchestrator marks the service "live", which is what surfaces as
    ``health_status = "live"`` on its model templates. Requires the ``X-Service-Key``
    header. Returns 404 if the service id is unknown.
    """
    try:
        return orchestrator.ping(service_id)
    except ServiceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e


@router.get(
    "",
    response_model=ServiceListResponse,
    response_model_exclude_none=True,
    summary="Browse currently live model services",
)
def list_services(
    orchestrator: Orchestrator = Depends(get_orchestrator),
) -> ServiceListResponse:
    """List every CHAPKit model service currently registered, so operators can see at a glance what compute is available to route work to."""
    return orchestrator.get_all()


@router.get(
    "/{service_id}",
    response_model=ServiceDetail,
    response_model_exclude_none=True,
    summary="Inspect one registered service",
)
def get_service(
    service_id: str,
    orchestrator: Orchestrator = Depends(get_orchestrator),
) -> ServiceDetail:
    """Look up everything the orchestrator knows about a single service — its declared info, the URL it is reachable at, and when it last pinged.

    Used to diagnose registration issues or populate a service-detail panel. Returns
    404 if the service id is unknown.
    """
    try:
        return orchestrator.get(service_id)
    except ServiceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e


@router.delete(
    "/{service_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Off-board a model service",
)
def deregister_service(
    service_id: str,
    orchestrator: Orchestrator = Depends(get_orchestrator),
    _: str = Depends(verify_service_key),
) -> None:
    """Remove a service from the registry — used by the service itself on graceful shutdown, or by an operator forcing eviction.

    Templates produced by the service stay in the database (so historical backtests
    still resolve) but lose their ``health_status = "live"`` marker. Requires the
    ``X-Service-Key`` header. Returns 204 on success and 404 if the service id is
    unknown.
    """
    try:
        orchestrator.deregister(service_id)
    except ServiceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e
