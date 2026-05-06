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


@router.post("/$register", response_model=RegistrationResponse)
def register_service(
    payload: RegistrationRequest,
    request: Request,
    orchestrator: Orchestrator = Depends(get_orchestrator),
    session: Session = Depends(get_session),
    _: str = Depends(verify_service_key),
) -> RegistrationResponse:
    """Register a new service with the orchestrator."""
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


@router.put("/{service_id}/$ping", response_model=PingResponse)
def ping_service(
    service_id: str,
    orchestrator: Orchestrator = Depends(get_orchestrator),
    _: str = Depends(verify_service_key),
) -> PingResponse:
    """Send a keepalive ping for a registered service."""
    try:
        return orchestrator.ping(service_id)
    except ServiceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e


@router.get("", response_model=ServiceListResponse, response_model_exclude_none=True)
def list_services(
    orchestrator: Orchestrator = Depends(get_orchestrator),
) -> ServiceListResponse:
    """List all registered services."""
    return orchestrator.get_all()


@router.get("/{service_id}", response_model=ServiceDetail, response_model_exclude_none=True)
def get_service(
    service_id: str,
    orchestrator: Orchestrator = Depends(get_orchestrator),
) -> ServiceDetail:
    """Get details of a specific registered service."""
    try:
        return orchestrator.get(service_id)
    except ServiceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e


@router.delete("/{service_id}", status_code=status.HTTP_204_NO_CONTENT)
def deregister_service(
    service_id: str,
    orchestrator: Orchestrator = Depends(get_orchestrator),
    _: str = Depends(verify_service_key),
) -> None:
    """Deregister a service."""
    try:
        orchestrator.deregister(service_id)
    except ServiceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e
