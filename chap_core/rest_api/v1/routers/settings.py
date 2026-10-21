from fastapi import APIRouter, Depends
from sqlmodel import Session

from chap_core.database.settings import Settings, get_backend_settings, get_settings_record
from chap_core.rest_api.v1.routers.dependencies import get_session

router = APIRouter(prefix="/settings", tags=["System"])


@router.get("", response_model=Settings, summary="Read the installation's benchmark and backtest defaults")
def read_settings(session: Session = Depends(get_session)):
    return get_backend_settings(session)


@router.put("", response_model=Settings, summary="Replace the installation's benchmark and backtest defaults")
def update_settings(settings: Settings, session: Session = Depends(get_session)):
    record = get_settings_record(session)
    record.values = settings.model_dump()
    session.add(record)
    session.commit()
    return settings
