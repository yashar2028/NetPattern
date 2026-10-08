"""Editor calls into a sandbox's engine session, forwarded to the worker (PLAN §3.3)."""

from typing import Any

import httpx
from fastapi import APIRouter, Body, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, not_found
from app.api.sandboxes import owned_sandbox
from app.core.config import settings
from app.core.errors import ApiError
from app.db.session import get_db
from app.models.user import User

router = APIRouter(prefix="/v1", tags=["editor"])

EDITOR_METHODS = {
    "schema",
    "palette",
    "validate_architecture",
    "inspect_model",
    "list_zoo",
    "model_card",
}
# The first call may build the environment and start PyTorch.
WORKER_TIMEOUT_SECONDS = 600


def worker_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=settings.WORKER_INTERNAL_URL, timeout=WORKER_TIMEOUT_SECONDS)


@router.post("/sandboxes/{sandbox_id}/session/{method}")
async def call_session(
    sandbox_id: str,
    method: str,
    params: dict[str, Any] = Body(default={}),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Any:
    """Runs in the sandbox's own engine version, so answers match what a run would do."""
    await owned_sandbox(db, sandbox_id, user)
    if method not in EDITOR_METHODS:
        raise not_found("Editor method")
    try:
        async with worker_client() as client:
            response = await client.post(f"/sessions/{sandbox_id}/{method}", json=params)
    except httpx.HTTPError as error:
        raise ApiError(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "worker_unavailable",
            "The sandbox worker is not reachable",
            str(error) or type(error).__name__,
        ) from error
    body = response.json()
    if response.status_code == 200:
        return body["result"]
    error = body.get("error", {})
    issues = (error.get("data") or {}).get("issues")
    raise ApiError(
        status.HTTP_422_UNPROCESSABLE_CONTENT,
        "session_error",
        "The sandbox could not answer",
        error.get("message"),
        issues,
    )
