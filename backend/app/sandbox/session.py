"""Client for an engine session process: JSON-RPC 2.0 over the process's stdin/stdout."""

from __future__ import annotations

import asyncio
import itertools
import json
import os
import signal
from pathlib import Path
from typing import Any

from app.sandbox.isolation import SandboxPolicy, sandbox_env, wrap
from app.sandbox.runner import STDOUT_LINE_LIMIT, runtime_command

STARTUP_TIMEOUT_SECONDS = 120


class SessionError(RuntimeError):
    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(message)
        self.code, self.data = code, data


class SessionClient:
    def __init__(self, env_path: Path, home: Path, policy: SandboxPolicy) -> None:
        self.env_path = env_path
        self.home = home
        self.policy = policy
        self.process: asyncio.subprocess.Process | None = None
        self._ids = itertools.count(1)
        self._lock = asyncio.Lock()

    async def start(self) -> dict[str, Any]:
        self.home.mkdir(parents=True, exist_ok=True)
        self.policy.read_write = [*self.policy.read_write, self.home]
        self.policy.working_dir = self.home
        self.process = await asyncio.create_subprocess_exec(
            *wrap(runtime_command(self.env_path, "session"), self.policy),
            cwd=str(self.home),
            env=sandbox_env(self.env_path, self.home, self.policy.network),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=(self.home / "session.log").open("ab"),
            limit=STDOUT_LINE_LIMIT,
            start_new_session=True,
        )
        ready = await asyncio.wait_for(self._read_message(), timeout=STARTUP_TIMEOUT_SECONDS)
        if ready.get("method") != "ready":
            raise SessionError(-1, f"unexpected first message from session: {ready}")
        return ready.get("params", {})

    async def _read_message(self) -> dict[str, Any]:
        assert self.process is not None and self.process.stdout is not None
        line = await self.process.stdout.readline()
        if not line:
            raise SessionError(-1, "the session process exited")
        return json.loads(line)

    async def call(
        self, method: str, params: dict[str, Any] | None = None, timeout: float = 120
    ) -> Any:
        if self.process is None or self.process.returncode is not None:
            raise SessionError(-1, "the session is not running")
        async with self._lock:
            request_id = next(self._ids)
            message = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}}
            assert self.process.stdin is not None
            self.process.stdin.write((json.dumps(message) + "\n").encode())
            await self.process.stdin.drain()
            while True:
                reply = await asyncio.wait_for(self._read_message(), timeout=timeout)
                if reply.get("id") == request_id:
                    break
        if "error" in reply:
            error = reply["error"]
            raise SessionError(
                error.get("code", -1), error.get("message", "error"), error.get("data")
            )
        return reply.get("result")

    async def close(self) -> None:
        if self.process is None or self.process.returncode is not None:
            return
        try:
            await self.call("shutdown", timeout=10)
        except (SessionError, asyncio.TimeoutError, json.JSONDecodeError):
            pass
        try:
            await asyncio.wait_for(self.process.wait(), timeout=10)
        except asyncio.TimeoutError:
            os.killpg(self.process.pid, signal.SIGKILL)
            await self.process.wait()
