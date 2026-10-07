"""PostgreSQL job queue (PLAN P10): SKIP LOCKED claims, LISTEN/NOTIFY wake-ups.

The API (Phase 2) and the worker CLI enqueue jobs; workers claim them. A job row is
the source of truth, the NOTIFY is only a wake-up call: a job submitted while every
worker is down simply waits in the table.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import func, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.job import Job, JobEvent, JobStatus

JOBS_CHANNEL = "jobs"
EVENTS_CHANNEL = "job_events"
CANCEL_CHANNEL = "job_cancel"


async def notify(db: AsyncSession, channel: str, payload: str) -> None:
    await db.execute(
        text("SELECT pg_notify(:channel, :payload)"), {"channel": channel, "payload": payload}
    )


async def enqueue(
    db: AsyncSession,
    kind: str,
    payload: dict[str, Any],
    environment: dict[str, Any] | None = None,
    hardware_tier: str = "cpu",
    priority: int = 0,
) -> Job:
    job = Job(
        kind=kind,
        payload=payload,
        environment=environment or {},
        hardware_tier=hardware_tier,
        priority=priority,
        status=JobStatus.QUEUED,
    )
    db.add(job)
    await db.flush()
    await notify(db, JOBS_CHANNEL, job.id)
    await db.commit()
    return job


async def claim_next(db: AsyncSession, worker_id: str, tiers: list[str]) -> Job | None:
    """Atomically move the next queued job for these tiers to 'running'."""
    candidate = (
        select(Job.id)
        .where(Job.status == JobStatus.QUEUED, Job.hardware_tier.in_(tiers))
        .order_by(Job.priority.desc(), Job.created_at)
        .limit(1)
        .with_for_update(skip_locked=True)
        .scalar_subquery()
    )
    now = datetime.now(UTC)
    claimed = await db.execute(
        update(Job)
        .where(Job.id == candidate)
        .values(
            status=JobStatus.RUNNING,
            worker_id=worker_id,
            started_at=now,
            heartbeat_at=now,
            attempts=Job.attempts + 1,
        )
        .returning(Job.id)
    )
    job_id = claimed.scalar_one_or_none()
    await db.commit()
    if job_id is None:
        return None
    return await db.get(Job, job_id, populate_existing=True)


async def heartbeat(db: AsyncSession, job_id: str) -> bool:
    """Refresh the heartbeat; returns True when cancellation was requested."""
    result = await db.execute(
        update(Job)
        .where(Job.id == job_id)
        .values(heartbeat_at=datetime.now(UTC))
        .returning(Job.cancel_requested)
    )
    cancel = bool(result.scalar_one_or_none())
    await db.commit()
    return cancel


async def finish(
    db: AsyncSession,
    job_id: str,
    status: JobStatus,
    result: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    await db.execute(
        update(Job)
        .where(Job.id == job_id)
        .values(status=status, result=result, error=error, finished_at=datetime.now(UTC))
    )
    await notify(db, EVENTS_CHANNEL, job_id)
    await db.commit()


async def requeue(db: AsyncSession, job_id: str, reason: str) -> None:
    await db.execute(
        update(Job)
        .where(Job.id == job_id)
        .values(
            status=JobStatus.QUEUED,
            worker_id=None,
            error=reason,
            started_at=None,
            heartbeat_at=None,
        )
    )
    await notify(db, JOBS_CHANNEL, job_id)
    await db.commit()


async def request_cancel(db: AsyncSession, job_id: str) -> JobStatus | None:
    """Queued jobs are cancelled at once; running ones are asked to stop."""
    job = await db.get(Job, job_id, populate_existing=True)
    if job is None:
        return None
    if job.status == JobStatus.QUEUED:
        job.status, job.finished_at = JobStatus.CANCELLED, datetime.now(UTC)
    elif job.status == JobStatus.RUNNING:
        job.cancel_requested = True
        await notify(db, CANCEL_CHANNEL, job_id)
    await db.commit()
    return job.status


async def recover_stale(db: AsyncSession, stale_seconds: int, max_attempts: int) -> list[str]:
    """Running jobs whose worker stopped heart-beating: retry, or fail after max attempts."""
    cutoff = datetime.now(UTC) - timedelta(seconds=stale_seconds)
    stale = (
        (
            await db.execute(
                select(Job)
                .where(Job.status == JobStatus.RUNNING, Job.heartbeat_at < cutoff)
                .with_for_update(skip_locked=True)
            )
        )
        .scalars()
        .all()
    )
    for job in stale:
        if job.attempts < max_attempts:
            job.status, job.worker_id, job.error = (
                JobStatus.QUEUED,
                None,
                "worker stopped; retrying",
            )
        else:
            job.status, job.finished_at = JobStatus.FAILED, datetime.now(UTC)
            job.error = "worker stopped while running the job"
    if stale:
        await notify(db, JOBS_CHANNEL, "recovered")
    await db.commit()
    return [job.id for job in stale]


async def add_event(
    db: AsyncSession, job_id: str, seq: int, event_type: str, payload: dict[str, Any]
) -> None:
    db.add(JobEvent(job_id=job_id, seq=seq, type=event_type, payload=payload))
    await notify(db, EVENTS_CHANNEL, job_id)
    await db.commit()


async def events_after(
    db: AsyncSession, job_id: str, after_seq: int, limit: int = 500
) -> list[JobEvent]:
    result = await db.execute(
        select(JobEvent)
        .where(JobEvent.job_id == job_id, JobEvent.seq > after_seq)
        .order_by(JobEvent.seq)
        .limit(limit)
    )
    return list(result.scalars().all())


async def last_seq(db: AsyncSession, job_id: str) -> int:
    value = await db.scalar(select(func.max(JobEvent.seq)).where(JobEvent.job_id == job_id))
    return int(value or 0)
