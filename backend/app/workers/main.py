"""Worker (runtime host) entrypoint: `python -m app.workers.main`.

The worker is the only process that starts sandbox processes. It claims queued jobs
for its hardware tiers, runs each in a pinned, isolated environment, streams events
into PostgreSQL, keeps a heartbeat, and honours cancellation.
"""

import asyncio
import logging
import shutil
import signal
import time

import asyncpg
from sqlalchemy.engine import make_url

from app import __version__
from app.core.config import settings
from app.db.session import AsyncSessionLocal
from app.models.job import Job, JobStatus
from app.workers import queue
from app.workers.handlers import JobHandler

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


class Worker:
    def __init__(self, stop: asyncio.Event) -> None:
        self.stop = stop
        self.wake = asyncio.Event()
        self.handler = JobHandler()
        self.current_job: str | None = None
        self.cancel_requested = False

    def _on_jobs(self, connection, pid, channel, payload) -> None:
        self.wake.set()

    def _on_cancel(self, connection, pid, channel, payload) -> None:
        if payload == self.current_job:
            self.cancel_requested = True

    async def run(self) -> None:
        nix_version = await detect_nix_version()
        listener = await connect_with_retry(asyncpg_dsn(settings.DATABASE_URL))
        await listener.add_listener(queue.JOBS_CHANNEL, self._on_jobs)
        await listener.add_listener(queue.CANCEL_CHANNEL, self._on_cancel)
        await self._recover()
        logger.info(
            "Worker %s (%s) ready: tiers=%s, isolation=%s, nix: %s",
            __version__,
            settings.WORKER_ID,
            ",".join(settings.worker_tiers),
            settings.SANDBOX_ISOLATION,
            nix_version or "not found",
        )
        last_recovery = time.monotonic()
        try:
            while not self.stop.is_set():
                async with AsyncSessionLocal() as db:
                    job = await queue.claim_next(db, settings.WORKER_ID, settings.worker_tiers)
                if job is not None:
                    await self._process(job)
                    continue
                if time.monotonic() - last_recovery > RECOVERY_INTERVAL_SECONDS:
                    await self._recover()
                    last_recovery = time.monotonic()
                self.wake.clear()
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
            await listener.close()
            logger.info("Worker stopped")

    async def _recover(self) -> None:
        async with AsyncSessionLocal() as db:
            recovered = await queue.recover_stale(
                db, settings.JOB_STALE_SECONDS, settings.JOB_MAX_ATTEMPTS
            )
        if recovered:
            logger.warning("Recovered orphaned job(s): %s", ", ".join(recovered))

    async def _process(self, job: Job) -> None:
        self.current_job, self.cancel_requested = job.id, False
        logger.info(
            "Claimed %s (%s, tier %s, attempt %s)",
            job.id,
            job.kind,
            job.hardware_tier,
            job.attempts,
        )

        async def beat() -> None:
            while True:
                await asyncio.sleep(CANCEL_CHECK_SECONDS)
                async with AsyncSessionLocal() as db:
                    if await queue.heartbeat(db, job.id):
                        self.cancel_requested = True

        beat_task = asyncio.create_task(beat())
        started = time.monotonic()
        try:
            async with AsyncSessionLocal() as db:
                job = await db.get(Job, job.id)
                status, result, error = await self.handler.run(
                    db, job, should_cancel=lambda: self.cancel_requested or self.stop.is_set()
                )
            async with AsyncSessionLocal() as db:
                if (
                    self.stop.is_set()
                    and not self.cancel_requested
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
            self.current_job = None


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
