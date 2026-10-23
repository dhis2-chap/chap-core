from fastapi import APIRouter, Depends

from chap_core.database.settings import Settings, SettingsManager
from chap_core.rest_api.v1.routers.dependencies import get_settings_manager

router = APIRouter(prefix="/settings", tags=["System"])


@router.get("", response_model=Settings, summary="Read the installation's benchmark and backtest defaults")
def read_settings(settings_manager: SettingsManager = Depends(get_settings_manager)):
    return settings_manager.get()


@router.put("", response_model=Settings, summary="Replace the installation's benchmark and backtest defaults")
def update_settings(settings: Settings, settings_manager: SettingsManager = Depends(get_settings_manager)):
    return settings_manager.update(settings)
