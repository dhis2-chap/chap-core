from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from chap_core.rest_api.v1.rest_api import app as v1_app
from chap_core.rest_api.v2.rest_api import app as v2_app

app = FastAPI(title="CHAP Core API")

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
