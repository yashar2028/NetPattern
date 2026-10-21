"""Calls from the API to a worker's internal API (editor sessions, deployments).

With several workers (PLAN Phase 8) the API picks one from the workers table: rendezvous hashing
keeps a key (a sandbox's editor session, a deployment) on the same live worker, and moves it
only when that worker disappears. Without any registered worker (tests, an old single setup) the
API uses WORKER_INTERNAL_URL.
"""

import hashlib
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
from fastapi import status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.errors import ApiError
from app.models.worker import WorkerNode

# The first call may build the environment and start PyTorch.
WORKER_TIMEOUT_SECONDS = 600
# A worker that has not reported for this long is treated as gone.
LIVE_SECONDS = 90


def worker_client(base_url: str | None = None) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=base_url or settings.WORKER_INTERNAL_URL, timeout=WORKER_TIMEOUT_SECONDS
    )


def unavailable(error: Exception) -> ApiError:
    return ApiError(
        status.HTTP_503_SERVICE_UNAVAILABLE,
        "worker_unavailable",
        "The sandbox worker is not reachable",
        str(error) or type(error).__name__,
    )


async def post(client: httpx.AsyncClient, path: str, payload: Any = None) -> tuple[bool, Any]:
    """(True, result) or (False, {code, message, data}) from the worker."""
    try:
        response = await client.post(path, json=payload if payload is not None else {})
    except httpx.HTTPError as error:
        raise unavailable(error) from error
    body = response.json()
    if response.status_code == 200:
        return True, body["result"]
    return False, body.get("error", {})


async def live_workers(db: AsyncSession) -> list[WorkerNode]:
    cutoff = datetime.now(UTC) - timedelta(seconds=LIVE_SECONDS)
    return list(
        (await db.execute(select(WorkerNode).where(WorkerNode.last_seen_at >= cutoff))).scalars()
    )


async def pick(
    db: AsyncSession, key: str, tier: str | None = None, account_id: str | None = None
) -> tuple[str | None, str]:
    """(worker id, url) for this key. Dedicated accounts use their own machine when it is up."""
    rows = await live_workers(db)
    if tier is not None:
        rows = [w for w in rows if tier in w.tiers]
    own = [w for w in rows if account_id and w.account_id == account_id]
    rows = own or [w for w in rows if w.account_id is None]
    if not rows:
        return None, settings.WORKER_INTERNAL_URL
    best = max(rows, key=lambda w: hashlib.sha256(f"{key}:{w.id}".encode()).digest())
    return best.id, best.url


async def url_of(db: AsyncSession, worker_id: str | None) -> str | None:
    """The URL of this worker if it is still live."""
    if worker_id is None:
        return None
    for row in await live_workers(db):
        if row.id == worker_id:
            return row.url
    return None
