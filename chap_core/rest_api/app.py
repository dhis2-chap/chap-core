from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from chap_core.rest_api.v1.rest_api import app as v1_app
from chap_core.rest_api.v2.rest_api import app as v2_app

app = FastAPI(title="CHAP Core API", docs_url=None, redoc_url=None, openapi_url=None)

origins = [
    "*",
    "http://localhost:3000",
    "localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/v1", v1_app)
app.mount("/v2", v2_app)


@app.get("/health")
async def root_health():
    return {"status": "healthy"}


@app.get("/docs", include_in_schema=False)
async def docs_redirect():
    return RedirectResponse(url="/v1/docs")


@app.get("/redoc", include_in_schema=False)
async def redoc_redirect():
    return RedirectResponse(url="/v1/redoc")


@app.get("/openapi.json", include_in_schema=False)
async def openapi_redirect():
    return RedirectResponse(url="/v1/openapi.json")
