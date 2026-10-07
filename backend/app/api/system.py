"""System endpoints: liveness/readiness and service info."""

import logging

from fastapi import APIRouter, Depends, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app import __version__
from app.core.config import settings
from app.core.errors import ApiError
from app.db.session import get_db
from app.schemas.system import HealthResponse, InfoResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
async def health(db: AsyncSession = Depends(get_db)) -> HealthResponse:
    """Report healthy only when the database answers."""
    try:
        await db.execute(text("SELECT 1"))
    except Exception as exc:
        logger.warning("Health check failed: database unavailable (%s)", exc)
        raise ApiError(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "database_unavailable",
            "Database unavailable",
            "The API is running but cannot reach PostgreSQL.",
        ) from exc
    return HealthResponse(status="ok", database="ok", version=__version__)


@router.get("/v1/info", response_model=InfoResponse)
async def info() -> InfoResponse:
    return InfoResponse(
        name=settings.APP_NAME,
        version=__version__,
        api_version="v1",
        environment=settings.APP_ENV,
    )
