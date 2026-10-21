"""Worker (runtime host) entrypoint: `python -m app.workers.main`.

The worker is the only process that starts sandbox processes. It claims queued jobs whose
bubble (hardware tier) fits its free capacity, runs several at once up to WORKER_MAX_JOBS,
each in a pinned, isolated environment on its own CPU cores and GPUs, streams events into
PostgreSQL, keeps a heartbeat per job, and honours cancellation.
"""

import asyncio
import contextlib
import functools
import logging
import shutil
import signal
import time
from datetime import UTC, datetime

import asyncpg
import uvicorn
from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import make_url

from app import __version__
from app.core.config import settings
from app.core.tiers import tiers
from app.db.session import AsyncSessionLocal
from app.models.job import Job, JobStatus
from app.models.user import User
from app.models.worker import WorkerNode
from app.workers import queue
from app.workers.capacity import Allocation, Capacity
from app.workers.deployments import DeploymentManager
from app.workers.handlers import JobHandler
from app.workers.sessions import SessionManager, internal_api

logger = logging.getLogger("netpattern.worker")

CONNECT_ATTEMPTS = 20
CONNECT_RETRY_SECONDS = 3
CANCEL_CHECK_SECONDS = 5
RECOVERY_INTERVAL_SECONDS = 60


def asyncpg_dsn(database_url: str) -> str:
    """Turn the SQLAlchemy URL (`postgresql+asyncpg://…`) into a plain asyncpg DSN."""
    return make_url(database_url).set(drivername="postgresql").render_as_string(hide_password=False)


async def detect_nix_version() -> str | None:
    nix = shutil.which("nix")
    if nix is None:
        return None
    process = await asyncio.create_subprocess_exec(
        nix,
        "--version",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    stdout, _ = await process.communicate()
    return stdout.decode().strip() if process.returncode == 0 else None


async def connect_with_retry(dsn: str) -> asyncpg.Connection:
    for attempt in range(1, CONNECT_ATTEMPTS + 1):
        try:
            return await asyncpg.connect(dsn)
        except (OSError, asyncpg.PostgresError) as exc:
            if attempt == CONNECT_ATTEMPTS:
                raise
            logger.warning(
                "Database not reachable yet (attempt %s/%s): %s", attempt, CONNECT_ATTEMPTS, exc
            )
            await asyncio.sleep(CONNECT_RETRY_SECONDS)
    raise RuntimeError("unreachable")


class _InternalServer(uvicorn.Server):
    """uvicorn inside the worker's own event loop; the worker handles the signals."""

    @contextlib.contextmanager
    def capture_signals(self):
        yield


class Worker:
    def __init__(self, stop: asyncio.Event) -> None:
        self.stop = stop
        self.wake = asyncio.Event()
        self.handler = JobHandler()
        self.sessions = SessionManager(self.handler.manager)
        self.deployments = DeploymentManager(self.handler.manager)
        self.capacity = Capacity.detect(
            settings.WORKER_CPUS,
            settings.WORKER_MEMORY_GB,
            settings.WORKER_GPUS,
            settings.WORKER_MAX_JOBS,
        )
        known = tiers()
        # Only tiers whose bubble fits this machine at all; others would wait forever here.
        self.served = [
            t for t in settings.worker_tiers if t in known and self.capacity.can_ever_run(known[t])
        ]
        self.account_id: str | None = None
        self.running: dict[str, asyncio.Task] = {}
        self.cancels: set[str] = set()

    async def _serve_sessions(self) -> None:
        server = _InternalServer(
            uvicorn.Config(
                internal_api(self.sessions, self.deployments),
                host="0.0.0.0",
                port=settings.WORKER_INTERNAL_PORT,
                log_level="warning",
            )
        )
        serving = asyncio.create_task(server.serve())
        restarting = asyncio.create_task(self.deployments.start_dedicated())
        try:
            while not self.stop.is_set():
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self.stop.wait(), timeout=60)
                await self.sessions.stop_idle()
                await self.deployments.stop_idle()
                await self.deployments.meter()
        finally:
            server.should_exit = True
            restarting.cancel()
            await serving
            await self.sessions.stop_all()
            await self.deployments.stop_all()

    def _on_jobs(self, connection, pid, channel, payload) -> None:
        self.wake.set()

    def _on_cancel(self, connection, pid, channel, payload) -> None:
        if payload in self.running:
            self.cancels.add(payload)

    async def _resolve_account(self) -> str | None:
        """WORKER_ACCOUNT (an email) makes this a dedicated machine for that account."""
        if not settings.WORKER_ACCOUNT:
            return None
        async with AsyncSessionLocal() as db:
            user_id = await db.scalar(
                select(User.id).where(User.email == settings.WORKER_ACCOUNT.strip().lower())
            )
        if user_id is None:
            raise SystemExit(f"WORKER_ACCOUNT {settings.WORKER_ACCOUNT} is not a known account")
        return user_id

    async def run(self) -> None:
        nix_version = await detect_nix_version()
        listener = await connect_with_retry(asyncpg_dsn(settings.DATABASE_URL))
        await listener.add_listener(queue.JOBS_CHANNEL, self._on_jobs)
        await listener.add_listener(queue.CANCEL_CHANNEL, self._on_cancel)
        self.account_id = await self._resolve_account()
        await self._recover()
        skipped = sorted(set(settings.worker_tiers) - set(self.served))
        logger.info(
            "Worker %s (%s) ready: tiers=%s%s, capacity=%s, %s, isolation=%s, nix: %s",
            __version__,
            settings.WORKER_ID,
            ",".join(self.served),
            f" (not on this machine: {','.join(skipped)})" if skipped else "",
            self.capacity.describe(),
            f"dedicated to {settings.WORKER_ACCOUNT}" if self.account_id else "shared",
            settings.SANDBOX_ISOLATION,
            nix_version or "not found",
        )
        last_recovery = time.monotonic()
        sessions_task = asyncio.create_task(self._serve_sessions())
        registry_task = asyncio.create_task(self._register())
        try:
            while not self.stop.is_set():
                # Clear first: a NOTIFY that arrives while claiming must wake the next round.
                self.wake.clear()
                await self._fill()
                if time.monotonic() - last_recovery > RECOVERY_INTERVAL_SECONDS:
                    await self._recover()
                    last_recovery = time.monotonic()
                waiters = [
                    asyncio.create_task(self.wake.wait()),
                    asyncio.create_task(self.stop.wait()),
                ]
                await asyncio.wait(
                    waiters,
                    timeout=settings.WORKER_POLL_SECONDS,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for waiter in waiters:
                    waiter.cancel()
        finally:
            self.stop.set()
            if self.running:
                await asyncio.gather(*self.running.values(), return_exceptions=True)
            registry_task.cancel()
            await self._unregister()
            await sessions_task
            await listener.close()
            logger.info("Worker stopped")

    async def _register(self) -> None:
        """Tell the API (workers table) where this worker is and what it serves."""
        url = settings.WORKER_ADVERTISE_URL or settings.WORKER_INTERNAL_URL
        while not self.stop.is_set():
            values = {
                "url": url,
                "tiers": self.served,
                "capacity": self.capacity.describe(),
                "account_id": self.account_id,
                "version": __version__,
                "running": len(self.running),
                "last_seen_at": datetime.now(UTC),
            }
            try:
                async with AsyncSessionLocal() as db:
                    await db.execute(
                        insert(WorkerNode)
                        .values(id=settings.WORKER_ID, **values)
                        .on_conflict_do_update(index_elements=[WorkerNode.id], set_=values)
                    )
                    await db.commit()
            except Exception:  # noqa: BLE001 - the registry must never stop the worker
                logger.exception("Could not update the worker registry")
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self.stop.wait(), timeout=settings.WORKER_HEARTBEAT_SECONDS)

    async def _unregister(self) -> None:
        with contextlib.suppress(Exception):
            async with AsyncSessionLocal() as db:
                await db.execute(delete(WorkerNode).where(WorkerNode.id == settings.WORKER_ID))
                await db.commit()

    async def _fill(self) -> None:
        """Claim jobs while a slot is free and some served bubble still fits."""
        known = tiers()
        while not self.stop.is_set():
            fitting = [t for t in self.served if self.capacity.fits(known[t])]
            if not fitting:
                return
            async with AsyncSessionLocal() as db:
                job = await queue.claim_next(db, settings.WORKER_ID, fitting, self.account_id)
            if job is None:
                return
            allocation = self.capacity.allocate(known[job.hardware_tier])
            task = asyncio.create_task(self._process(job, allocation))
            self.running[job.id] = task
            task.add_done_callback(functools.partial(self._finished, job.id, allocation))

    def _finished(self, job_id: str, allocation: Allocation, task: asyncio.Task) -> None:
        self.capacity.release(allocation)
        self.running.pop(job_id, None)
        self.cancels.discard(job_id)
        self.wake.set()  # capacity is free again

    async def _recover(self) -> None:
        async with AsyncSessionLocal() as db:
            recovered = await queue.recover_stale(
                db, settings.JOB_STALE_SECONDS, settings.JOB_MAX_ATTEMPTS
            )
        if recovered:
            logger.warning("Recovered orphaned job(s): %s", ", ".join(recovered))

    async def _process(self, job: Job, allocation: Allocation) -> None:
        logger.info(
            "Claimed %s (%s, tier %s, cpus %s, gpus %s, attempt %s)",
            job.id,
            job.kind,
            job.hardware_tier,
            ",".join(map(str, allocation.cpus)) or "-",
            ",".join(map(str, allocation.gpus)) or "-",
            job.attempts,
        )

        async def beat() -> None:
            while True:
                await asyncio.sleep(CANCEL_CHECK_SECONDS)
                async with AsyncSessionLocal() as db:
                    if await queue.heartbeat(db, job.id):
                        self.cancels.add(job.id)

        beat_task = asyncio.create_task(beat())
        started = time.monotonic()
        try:
            async with AsyncSessionLocal() as db:
                job = await db.get(Job, job.id)
                status, result, error = await self.handler.run(
                    db,
                    job,
                    should_cancel=lambda: job.id in self.cancels or self.stop.is_set(),
                    resources=allocation,
                )
            async with AsyncSessionLocal() as db:
                if (
                    self.stop.is_set()
                    and job.id not in self.cancels
                    and status == JobStatus.CANCELLED
                ):
                    # The worker is shutting down: give the job back instead of losing it.
                    if job.attempts < settings.JOB_MAX_ATTEMPTS:
                        await queue.requeue(db, job.id, "worker stopped; will retry")
                        logger.warning("Requeued %s because the worker is stopping", job.id)
                        return
                    status, error = JobStatus.FAILED, "worker stopped while running the job"
                await queue.finish(db, job.id, status, result, error)
            logger.info(
                "Finished %s: %s in %.1fs%s",
                job.id,
                status.value,
                time.monotonic() - started,
                f" ({error.splitlines()[0]})" if error else "",
            )
        except Exception as error:  # noqa: BLE001 - a broken job must never stop the worker
            logger.exception("Job %s crashed the handler", job.id)
            async with AsyncSessionLocal() as db:
                await queue.finish(db, job.id, JobStatus.FAILED, None, f"internal error: {error}")
        finally:
            beat_task.cancel()


async def _main() -> None:
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for stop_signal in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(stop_signal, stop.set)
    await Worker(stop).run()


def main() -> None:
    logging.basicConfig(
        level=settings.LOG_LEVEL.upper(),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    asyncio.run(_main())


if __name__ == "__main__":
    main()
