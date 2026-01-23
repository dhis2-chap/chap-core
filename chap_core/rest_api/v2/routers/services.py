from fastapi import APIRouter, Depends, HTTPException, status

from chap_core.rest_api.services.orchestrator import Orchestrator, ServiceNotFoundError
from chap_core.rest_api.services.schemas import (
    PingResponse,
    RegistrationPayload,
    RegistrationResponse,
    ServiceDetail,
    ServiceListResponse,
)
from chap_core.rest_api.v2.dependencies import get_orchestrator, verify_service_key

router = APIRouter(prefix="/services", tags=["services"])


@router.post("/$register", response_model=RegistrationResponse)
def register_service(
    payload: RegistrationPayload,
    orchestrator: Orchestrator = Depends(get_orchestrator),
    _: str = Depends(verify_service_key),
) -> RegistrationResponse:
    """Register a new service with the orchestrator."""
    return orchestrator.register(payload)


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
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.get("", response_model=ServiceListResponse)
def list_services(
    orchestrator: Orchestrator = Depends(get_orchestrator),
) -> ServiceListResponse:
    """List all registered services."""
    return orchestrator.get_all()


@router.get("/{service_id}", response_model=ServiceDetail)
def get_service(
    service_id: str,
    orchestrator: Orchestrator = Depends(get_orchestrator),
) -> ServiceDetail:
    """Get details of a specific registered service."""
    try:
        return orchestrator.get(service_id)
    except ServiceNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


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
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
