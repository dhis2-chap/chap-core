from fastapi import APIRouter

from chap_core.rest_api.v2.routers import proxy, services

router = APIRouter()
router.include_router(services.router)
router.include_router(proxy.router)
