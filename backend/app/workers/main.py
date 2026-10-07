"""Worker (runtime host) entrypoint: `python -m app.workers.main`.

The worker is the only process that starts sandbox processes. Phase 0 keeps it
minimal: connect to PostgreSQL, listen on the `jobs` channel, log a heartbeat
and stop cleanly on SIGTERM. Claiming jobs, building Nix environments and
running sandboxes arrive in Phase 1.
"""

import asyncio
import logging
import shutil
import signal

import asyncpg
from sqlalchemy.engine import make_url

from app import __version__
from app.core.config import settings

logger = logging.getLogger("netpattern.worker")

JOBS_CHANNEL = "jobs"
CONNECT_ATTEMPTS = 20
CONNECT_RETRY_SECONDS = 3


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


def _on_jobs_notification(
    connection: asyncpg.Connection, pid: int, channel: str, payload: str
) -> None:
    logger.info("Notification on '%s': %s (job handling arrives in Phase 1)", channel, payload)


async def run_worker(stop: asyncio.Event) -> None:
    nix_version = await detect_nix_version()
    connection = await connect_with_retry(asyncpg_dsn(settings.DATABASE_URL))
    await connection.add_listener(JOBS_CHANNEL, _on_jobs_notification)
    logger.info(
        "Worker %s ready: listening on '%s', nix: %s",
        __version__,
        JOBS_CHANNEL,
        nix_version or "not found",
    )

    try:
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=settings.WORKER_HEARTBEAT_SECONDS)
            except TimeoutError:
                await connection.execute("SELECT 1")
                logger.info("Heartbeat: database connection alive")
    finally:
        await connection.remove_listener(JOBS_CHANNEL, _on_jobs_notification)
        await connection.close()
        logger.info("Worker stopped")


async def _main() -> None:
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for stop_signal in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(stop_signal, stop.set)
    await run_worker(stop)


def main() -> None:
    logging.basicConfig(
        level=settings.LOG_LEVEL.upper(),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    asyncio.run(_main())


if __name__ == "__main__":
    main()
