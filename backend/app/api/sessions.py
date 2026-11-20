"""Editor calls into a sandbox's engine session, forwarded to the worker (PLAN §3.3)."""

from typing import Any

from fastapi import APIRouter, Body, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, not_found
from app.api.sandboxes import owned_sandbox
from app.core.errors import ApiError
from app.db.session import get_db
from app.models.user import User
from app.services import worker
from app.services.worker import worker_client

router = APIRouter(prefix="/v1", tags=["editor"])

EDITOR_METHODS = {
    "schema",
    "palette",
    "validate_architecture",
    "inspect_model",
    "list_zoo",
    "model_card",
}


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
    account = user.id if user.dedicated_worker else None
    _, url = await worker.pick(db, f"session:{sandbox_id}", account_id=account)
    async with worker_client(url) as client:
        ok, answer = await worker.post(client, f"/sessions/{sandbox_id}/{method}", params)
    if ok:
        return answer
    error = answer
    issues = (error.get("data") or {}).get("issues")
    raise ApiError(
        status.HTTP_422_UNPROCESSABLE_CONTENT,
        "session_error",
        "The sandbox could not answer",
        error.get("message"),
        issues,
    )
