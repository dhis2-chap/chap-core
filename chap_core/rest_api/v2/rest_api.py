from fastapi import APIRouter

from chap_core.rest_api.v2.routers import services

router = APIRouter()
router.include_router(services.router)
