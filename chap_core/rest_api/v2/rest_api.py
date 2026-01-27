from fastapi import FastAPI
from fastapi.responses import ORJSONResponse

from chap_core.rest_api.v2.routers import services

app = FastAPI(root_path="/v2", default_response_class=ORJSONResponse)
app.include_router(services.router)
