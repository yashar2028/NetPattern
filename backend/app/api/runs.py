"""Runs (jobs started through the API), their live events, and usage."""

import asyncio
import json

from fastapi import APIRouter, Depends, Query, Request, status
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, not_found
from app.api.sandboxes import owned_sandbox
from app.core.errors import ApiError
from app.core.pagination import DEFAULT_LIMIT, Page, paginate
from app.db.session import AsyncSessionLocal, get_db
from app.models.job import FINISHED_STATUSES, Job, JobStatus
from app.models.sandbox import Pipeline, PipelineVersion
from app.models.usage import UsageRecord
from app.models.user import User
from app.schemas.api import RunCreate, RunEventOut, RunOut, UsageOut
from app.services.runs import RunRequestError, resolve_datasets
from app.workers import queue

router = APIRouter(prefix="/v1", tags=["runs"])

STREAM_POLL_SECONDS = 1.0
STREAM_KEEPALIVE_SECONDS = 15.0


async def owned_run(db: AsyncSession, run_id: str, user: User) -> Job:
    job = await db.get(Job, run_id)
    if job is None or job.owner_id != user.id:
        raise not_found("Run")
    return job


@router.post(
    "/sandboxes/{sandbox_id}/runs", response_model=RunOut, status_code=status.HTTP_202_ACCEPTED
)
async def start_run(
    sandbox_id: str,
    body: RunCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Job:
    sandbox = await owned_sandbox(db, sandbox_id, user)
    version = await db.get(PipelineVersion, body.pipeline_version_id)
    pipeline = await db.get(Pipeline, version.pipeline_id) if version else None
    if version is None or pipeline is None or pipeline.sandbox_id != sandbox.id:
        raise not_found("Pipeline version")
    try:
        spec = await resolve_datasets(db, version.spec, user)
    except RunRequestError as error:
        raise ApiError(
            status.HTTP_422_UNPROCESSABLE_CONTENT, "invalid_run", "Cannot start the run", str(error)
        ) from error

    environment = {"template": sandbox.template}
    if sandbox.engine_snapshot:
        environment["engine_snapshot"] = sandbox.engine_snapshot
    return await queue.enqueue(
        db,
        body.kind,
        {"pipeline": spec},
        environment,
        hardware_tier=body.hardware_tier,
        owner_id=user.id,
        sandbox_id=sandbox.id,
        pipeline_version_id=version.id,
    )


@router.get("/runs", response_model=Page[RunOut])
async def list_runs(
    sandbox_id: str | None = None,
    status_filter: JobStatus | None = Query(None, alias="status"),
    limit: int = Query(DEFAULT_LIMIT),
    cursor: str | None = None,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    query = select(Job).where(Job.owner_id == user.id)
    if sandbox_id:
        query = query.where(Job.sandbox_id == sandbox_id)
    if status_filter:
        query = query.where(Job.status == status_filter)
    rows, next_cursor = await paginate(db, query, Job, limit, cursor)
    # Lists stay small: full results are on GET /v1/runs/{id}.
    return {
        "data": [RunOut.model_validate(r).model_copy(update={"result": None}) for r in rows],
        "next_cursor": next_cursor,
    }


@router.get("/runs/{run_id}", response_model=RunOut)
async def get_run(
    run_id: str, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
) -> Job:
    return await owned_run(db, run_id, user)


@router.post("/runs/{run_id}/cancel", response_model=RunOut)
async def cancel_run(
    run_id: str, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
) -> Job:
    job = await owned_run(db, run_id, user)
    await queue.request_cancel(db, job.id)
    return await db.get(Job, job.id, populate_existing=True)


@router.post("/runs/{run_id}/rerun", response_model=RunOut, status_code=status.HTTP_202_ACCEPTED)
async def rerun(
    run_id: str, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
) -> Job:
    """Run again with the same request and, when it exists, the exact same environment."""
    job = await owned_run(db, run_id, user)
    environment = (
        {"environment_id": job.environment_id} if job.environment_id else dict(job.environment)
    )
    return await queue.enqueue(
        db,
        job.kind,
        job.payload,
        environment,
        hardware_tier=job.hardware_tier,
        owner_id=user.id,
        sandbox_id=job.sandbox_id,
        pipeline_version_id=job.pipeline_version_id,
    )


@router.get("/runs/{run_id}/events", response_model=list[RunEventOut])
async def list_events(
    run_id: str,
    after: int = Query(0, ge=0),
    limit: int = Query(500, ge=1, le=2000),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    job = await owned_run(db, run_id, user)
    return await queue.events_after(db, job.id, after, limit)


@router.get("/runs/{run_id}/events/stream")
async def stream_events(
    run_id: str,
    request: Request,
    after: int = Query(0, ge=0),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """Server-Sent Events; reconnect with Last-Event-ID to resume (PLAN §8.7)."""
    job = await owned_run(db, run_id, user)
    start = int(request.headers.get("last-event-id") or after)

    async def events():
        seq, idle = start, 0.0
        while not await request.is_disconnected():
            async with AsyncSessionLocal() as session:
                batch = await queue.events_after(session, job.id, seq)
                current = await session.get(Job, job.id)
            for event in batch:
                seq = event.seq
                data = json.dumps(event.payload, default=str)
                yield f"id: {event.seq}\nevent: {event.type}\ndata: {data}\n\n"
            if current is None or (current.status in FINISHED_STATUSES and not batch):
                status_value = current.status.value if current else "deleted"
                yield f"event: end\ndata: {json.dumps({'status': status_value})}\n\n"
                return
            idle = 0.0 if batch else idle + STREAM_POLL_SECONDS
            if idle >= STREAM_KEEPALIVE_SECONDS:
                idle = 0.0
                yield ": keep-alive\n\n"
            await asyncio.sleep(STREAM_POLL_SECONDS)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/usage", response_model=UsageOut, tags=["usage"])
async def usage(
    user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
) -> UsageOut:
    """Compute time used, by hardware tier and sandbox (no prices yet)."""
    mine = UsageRecord.owner_id == user.id
    by_tier = dict(
        (
            await db.execute(
                select(UsageRecord.hardware_tier, func.sum(UsageRecord.seconds))
                .where(mine)
                .group_by(UsageRecord.hardware_tier)
            )
        ).all()
    )
    by_sandbox = dict(
        (
            await db.execute(
                select(UsageRecord.sandbox_id, func.sum(UsageRecord.seconds))
                .where(mine)
                .group_by(UsageRecord.sandbox_id)
            )
        ).all()
    )
    jobs = await db.scalar(select(func.count(UsageRecord.id)).where(mine)) or 0
    return UsageOut(
        total_seconds=round(sum(by_tier.values()), 2),
        jobs=jobs,
        by_tier={tier: round(seconds, 2) for tier, seconds in by_tier.items()},
        by_sandbox={
            str(sandbox or "none"): round(seconds, 2) for sandbox, seconds in by_sandbox.items()
        },
    )
