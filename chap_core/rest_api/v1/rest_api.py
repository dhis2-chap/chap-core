import logging

from fastapi import APIRouter

from chap_core.log_config import initialize_logging
from chap_core.rest_api.v1.routers import analytics, crud, visualization

from . import jobs

initialize_logging(True, "logs/rest_api.log")
logger = logging.getLogger(__name__)
logger.info("Logging initialized")

router = APIRouter()
router.include_router(crud.router)
router.include_router(analytics.router)
router.include_router(jobs.router)
router.include_router(visualization.router)
