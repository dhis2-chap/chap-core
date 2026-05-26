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
    summary="Register a chapkit service with the orchestrator",
)
def register_service(
    payload: RegistrationRequest,
    request: Request,
    orchestrator: Orchestrator = Depends(get_orchestrator),
    session: Session = Depends(get_session),
    _: str = Depends(verify_service_key),
) -> RegistrationResponse:
    """Register a new chapkit service and return the absolute ping URL the caller must keep alive.

    Eagerly syncs the registered service into the v1 model-templates and configured-models
    tables so it shows up immediately via the v1 CRUD endpoints. Requires the
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
    summary="Keep a registered service alive",
)
def ping_service(
    service_id: str,
    orchestrator: Orchestrator = Depends(get_orchestrator),
    _: str = Depends(verify_service_key),
) -> PingResponse:
    """Refresh the registry TTL for a registered service so the orchestrator does not evict it.

    Requires the ``X-Service-Key`` header. Returns 404 if the service id is unknown to
    the orchestrator.
    """
    try:
        return orchestrator.ping(service_id)
    except ServiceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e


@router.get(
    "",
    response_model=ServiceListResponse,
    response_model_exclude_none=True,
    summary="List registered services",
)
def list_services(
    orchestrator: Orchestrator = Depends(get_orchestrator),
) -> ServiceListResponse:
    """Return every service currently registered with the orchestrator and a count."""
    return orchestrator.get_all()


@router.get(
    "/{service_id}",
    response_model=ServiceDetail,
    response_model_exclude_none=True,
    summary="Get a registered service",
)
def get_service(
    service_id: str,
    orchestrator: Orchestrator = Depends(get_orchestrator),
) -> ServiceDetail:
    """Return the full detail (info, URL, last ping) for a single registered service.

    Returns 404 if the service id is unknown to the orchestrator.
    """
    try:
        return orchestrator.get(service_id)
    except ServiceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e


@router.delete(
    "/{service_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Deregister a service",
)
def deregister_service(
    service_id: str,
    orchestrator: Orchestrator = Depends(get_orchestrator),
    _: str = Depends(verify_service_key),
) -> None:
    """Remove a service from the orchestrator's registry. Requires the ``X-Service-Key`` header.

    Returns 204 No Content on success and 404 if the service id is unknown.
    """
    try:
        orchestrator.deregister(service_id)
    except ServiceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e
