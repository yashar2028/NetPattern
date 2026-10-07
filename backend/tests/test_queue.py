"""Queue behaviour against the real PostgreSQL (run inside docker compose)."""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import delete, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import settings
from app.models.job import Job, JobStatus
from app.workers import queue

pytestmark = pytest.mark.integration

TIER = "test-tier"


@pytest.fixture
async def sessions():
    engine = create_async_engine(settings.DATABASE_URL, poolclass=NullPool)
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    yield factory
    async with factory() as db:
        await db.execute(delete(Job).where(Job.hardware_tier == TIER))
        await db.commit()
    await engine.dispose()


async def test_jobs_are_claimed_once_in_priority_order(sessions):
    async with sessions() as db:
        low = await queue.enqueue(db, "smoke", {}, hardware_tier=TIER, priority=0)
        high = await queue.enqueue(db, "smoke", {}, hardware_tier=TIER, priority=5)

    async with sessions() as first, sessions() as second:
        claimed_a = await queue.claim_next(first, "worker-a", [TIER])
        claimed_b = await queue.claim_next(second, "worker-b", [TIER])
        claimed_c = await queue.claim_next(first, "worker-a", [TIER])

    assert claimed_a.id == high.id and claimed_a.status == JobStatus.RUNNING
    assert claimed_b.id == low.id and claimed_b.worker_id == "worker-b"
    assert claimed_c is None
    assert claimed_a.attempts == 1


async def test_cancel_queued_and_running_jobs(sessions):
    async with sessions() as db:
        queued = await queue.enqueue(db, "smoke", {}, hardware_tier=TIER)
        running = await queue.enqueue(db, "smoke", {}, hardware_tier=TIER, priority=1)
        await queue.claim_next(db, "worker-a", [TIER])

        assert await queue.request_cancel(db, queued.id) == JobStatus.CANCELLED
        assert await queue.request_cancel(db, running.id) == JobStatus.RUNNING
        assert await queue.heartbeat(db, running.id) is True


async def test_events_are_ordered_and_finish_records_the_result(sessions):
    async with sessions() as db:
        job = await queue.enqueue(db, "smoke", {}, hardware_tier=TIER)
        for seq, kind in enumerate(["run_started", "epoch_end", "run_completed"], start=1):
            await queue.add_event(db, job.id, seq, kind, {"n": seq})
        await queue.finish(db, job.id, JobStatus.COMPLETED, {"status": "completed"})

        events = await queue.events_after(db, job.id, 1)
        finished = await db.get(Job, job.id, populate_existing=True)

    assert [e.type for e in events] == ["epoch_end", "run_completed"]
    assert finished.status == JobStatus.COMPLETED and finished.result == {"status": "completed"}
    assert finished.finished_at is not None


async def test_stale_jobs_are_retried_then_failed(sessions):
    async with sessions() as db:
        job = await queue.enqueue(db, "smoke", {}, hardware_tier=TIER)
        await queue.claim_next(db, "worker-a", [TIER])
        old = datetime.now(UTC) - timedelta(minutes=10)
        await db.execute(update(Job).where(Job.id == job.id).values(heartbeat_at=old))
        await db.commit()

        assert job.id in await queue.recover_stale(db, stale_seconds=60, max_attempts=2)
        assert (await db.get(Job, job.id, populate_existing=True)).status == JobStatus.QUEUED

        await queue.claim_next(db, "worker-a", [TIER])
        await db.execute(update(Job).where(Job.id == job.id).values(heartbeat_at=old))
        await db.commit()
        await queue.recover_stale(db, stale_seconds=60, max_attempts=2)
        failed = await db.get(Job, job.id, populate_existing=True)

    assert failed.status == JobStatus.FAILED and failed.attempts == 2
