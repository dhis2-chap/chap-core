import logging

from fastapi import APIRouter

from chap_core.log_config import initialize_logging
from chap_core.rest_api.v1.routers import analytics, crud, legacy, visualization

from . import debug, jobs

initialize_logging(True, "logs/rest_api.log")
logger = logging.getLogger(__name__)
logger.info("Logging initialized")

router = APIRouter()
router.include_router(crud.router)
router.include_router(analytics.router)
router.include_router(debug.router)
router.include_router(jobs.router)
router.include_router(legacy.router)
router.include_router(visualization.router)


def get_openapi_schema():
    from chap_core.rest_api.app import app

    return app.openapi()


def main_backend(seed_data=None, auto_reload=False):
    import uvicorn

    from chap_core.database.database import create_db_and_tables
    from chap_core.rest_api.app import app
    from chap_core.rest_api.v1.routers.legacy import seed

    create_db_and_tables()

    if seed_data is not None:
        seed(seed_data)

    if auto_reload:
        app_path = "chap_core.rest_api.app:app"
        uvicorn.run(app_path, host="0.0.0.0", port=8000, reload=auto_reload)
    else:
        uvicorn.run(app, host="0.0.0.0", port=8000)
