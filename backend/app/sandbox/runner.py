"""Run one engine job in a sandbox and stream its events (PLAN §3.2, §4.4).

Contract (as in mcpquick, extended with streaming):
    request.json in -> JSONL events on stdout, read line by line -> result.json out
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.sandbox.isolation import SandboxPolicy, sandbox_env, wrap

logger = logging.getLogger(__name__)

STDOUT_LINE_LIMIT = 16 * 1024 * 1024
CANCEL_POLL_SECONDS = 1.0
# Created in the run directory to ask the engine to stop (see engine runtime.CANCEL_FILE).
CANCEL_FILE = "cancel.request"

OnEvent = Callable[[dict[str, Any]], Awaitable[None]]
ShouldCancel = Callable[[], bool]


@dataclass
class JobOutcome:
    exit_code: int | None
    result: dict[str, Any] | None
    cancelled: bool = False
    timed_out: bool = False


def runtime_command(env_path: Path, *args: str) -> list[str]:
    return [str(env_path / "bin" / "netpattern-runtime"), *args]


def _signal_group(process: asyncio.subprocess.Process, sig: int) -> None:
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        pass


async def run_engine_job(
    env_path: Path,
    run_dir: Path,
    request: dict[str, Any],
    on_event: OnEvent,
    should_cancel: ShouldCancel,
    policy: SandboxPolicy,
    timeout: float | None = None,
    result_name: str = "result.json",
) -> JobOutcome:
    run_dir.mkdir(parents=True, exist_ok=True)
    request_file = run_dir / result_name.replace("result", "request")
    result_file = run_dir / result_name
    request_file.write_text(json.dumps(request, indent=2), encoding="utf-8")
    result_file.unlink(missing_ok=True)
    cancel_file = run_dir / CANCEL_FILE
    cancel_file.unlink(missing_ok=True)

    command = runtime_command(
        env_path, "job", "--request", str(request_file), "--output", str(result_file)
    )
    policy.read_write = [*policy.read_write, run_dir]
    policy.working_dir = run_dir
    stderr_log = (run_dir / "stderr.log").open("ab")
    process = await asyncio.create_subprocess_exec(
        *wrap(command, policy),
        cwd=str(run_dir),
        env=sandbox_env(env_path, run_dir, policy.network),
        stdout=asyncio.subprocess.PIPE,
        stderr=stderr_log,
        limit=STDOUT_LINE_LIMIT,
        start_new_session=True,
    )

    outcome = JobOutcome(exit_code=None, result=None)
    deadline = time.monotonic() + (timeout or settings.SANDBOX_JOB_TIMEOUT_SECONDS)

    async def watchdog() -> None:
        # 1. ask politely (request file: the engine saves a checkpoint and exits)
        # 2. after half the grace period, SIGTERM; 3. after the full period, SIGKILL.
        asked_at: float | None = None
        terminated = False
        grace = settings.SANDBOX_CANCEL_GRACE_SECONDS
        while process.returncode is None:
            await asyncio.sleep(CANCEL_POLL_SECONDS)
            if asked_at is None:
                if should_cancel():
                    outcome.cancelled = True
                elif time.monotonic() > deadline:
                    outcome.timed_out = True
                if outcome.cancelled or outcome.timed_out:
                    cancel_file.touch()
                    asked_at = time.monotonic()
                continue
            waited = time.monotonic() - asked_at
            if waited > grace:
                _signal_group(process, signal.SIGKILL)
                return
            if waited > grace / 2 and not terminated:
                _signal_group(process, signal.SIGTERM)
                terminated = True

    watcher = asyncio.create_task(watchdog())
    try:
        assert process.stdout is not None
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            try:
                event = json.loads(text)
            except json.JSONDecodeError:
                stderr_log.write(f"[stdout] {text}\n".encode())
                continue
            if isinstance(event, dict) and "type" in event:
                await on_event(event)
        outcome.exit_code = await process.wait()
    finally:
        watcher.cancel()
        if process.returncode is None:
            _signal_group(process, signal.SIGKILL)
            await process.wait()
        stderr_log.close()

    if result_file.exists():
        try:
            outcome.result = json.loads(result_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("Invalid result file in %s", run_dir)
    return outcome


def stderr_tail(run_dir: Path, lines: int = 20) -> str:
    path = run_dir / "stderr.log"
    if not path.exists():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
