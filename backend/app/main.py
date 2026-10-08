"""FastAPI application entrypoint and top-level router wiring."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.auth import router as auth_router
from app.api.datasets import router as datasets_router
from app.api.runs import router as runs_router
from app.api.sandboxes import router as sandboxes_router
from app.api.sessions import router as sessions_router
from app.api.system import router as system_router
from app.core.config import settings
from app.core.errors import register_error_handlers
from app.core.request_id import REQUEST_ID_HEADER, RequestIdMiddleware

app = FastAPI(
    title="NetPattern API",
    version=__version__,
    openapi_url="/v1/openapi.json",
)

# Starlette wraps later middleware around earlier ones, so CORS stays outermost.
app.add_middleware(RequestIdMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[REQUEST_ID_HEADER],
)

register_error_handlers(app)

for router in (
    system_router,
    auth_router,
    sandboxes_router,
    sessions_router,
    datasets_router,
    runs_router,
):
    app.include_router(router)
