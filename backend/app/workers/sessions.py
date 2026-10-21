"""Editor sessions (PLAN §3.3, §4.4): one engine session process per sandbox.

The backend forwards editor calls (schema, validate_architecture, list_zoo, …) to the
worker's internal HTTP API; this manager starts the sandbox's session on first use,
in the sandbox's pinned environment and isolation, and stops it when idle.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.db.session import AsyncSessionLocal
from app.models.sandbox import Sandbox
from app.sandbox.environments import EnvironmentError_, EnvironmentManager
from app.sandbox.isolation import SandboxPolicy
from app.sandbox.session import SessionClient, SessionError
from app.services import storage

logger = logging.getLogger("netpattern.worker")

CALL_TIMEOUT_SECONDS = 120


class SessionManager:
    def __init__(self, manager: EnvironmentManager) -> None:
        self.manager = manager
        self.sessions: dict[str, SessionClient] = {}
        self.last_used: dict[str, float] = {}
        # The engine snapshot each running session was started with.
        self.snapshots: dict[str, str] = {}
        self.locks: dict[str, asyncio.Lock] = {}

    async def call(self, sandbox_id: str, method: str, params: dict[str, Any]) -> Any:
        async with self.locks.setdefault(sandbox_id, asyncio.Lock()):
            client = await self._session(sandbox_id)
            self.last_used[sandbox_id] = time.monotonic()
            try:
                return await client.call(method, params, timeout=CALL_TIMEOUT_SECONDS)
            except (asyncio.TimeoutError, OSError) as error:
                await self._stop(sandbox_id)  # a stuck session is restarted on the next call
                raise SessionError(-1, f"the session did not answer: {error!r}") from error

    async def _session(self, sandbox_id: str) -> SessionClient:
        async with AsyncSessionLocal() as db:
            sandbox = await db.get(Sandbox, sandbox_id)
            if sandbox is None:
                raise SessionError(-1, "sandbox not found")
            client = self.sessions.get(sandbox_id)
            running = (
                client is not None
                and client.process is not None
                and client.process.returncode is None
            )
            # After an engine upgrade the sandbox is unpinned: restart on the new engine.
            if running and self.snapshots.get(sandbox_id) == sandbox.engine_snapshot:
                return client
            if client is not None:
                await self._stop(sandbox_id)
            try:
                environment = await self.manager.ensure(
                    db, sandbox.template, [], sandbox.engine_snapshot
                )
            except EnvironmentError_ as error:
                raise SessionError(-1, str(error)) from error
            sandbox.pin(environment.engine_snapshot, environment.engine_version)
            await db.commit()
        client = SessionClient(
            Path(environment.store_path),
            storage.root() / "sessions" / sandbox_id,
            SandboxPolicy(read_only=[storage.cache_root()]),
        )
        await client.start()
        self.sessions[sandbox_id] = client
        self.snapshots[sandbox_id] = environment.engine_snapshot
        logger.info("Started editor session for %s in %s", sandbox_id, environment.id)
        return client

    async def _stop(self, sandbox_id: str) -> None:
        client = self.sessions.pop(sandbox_id, None)
        self.last_used.pop(sandbox_id, None)
        self.snapshots.pop(sandbox_id, None)
        if client is not None:
            await client.close()

    async def stop_idle(self) -> None:
        cutoff = time.monotonic() - settings.SANDBOX_SESSION_IDLE_SECONDS
        for sandbox_id in [s for s, used in self.last_used.items() if used < cutoff]:
            async with self.locks.setdefault(sandbox_id, asyncio.Lock()):
                await self._stop(sandbox_id)
                logger.info("Stopped idle editor session for %s", sandbox_id)

    async def stop_all(self) -> None:
        for sandbox_id in list(self.sessions):
            await self._stop(sandbox_id)


def _error(error: SessionError) -> JSONResponse:
    return JSONResponse(
        {"error": {"code": error.code, "message": str(error), "data": error.data}},
        status_code=422,
    )


def internal_api(sessions: SessionManager, deployments: Any = None) -> FastAPI:
    """Reachable only on the compose network; the backend checks ownership first."""
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.post("/sessions/{sandbox_id}/{method}")
    async def call(sandbox_id: str, method: str, params: dict[str, Any] = Body(default={})):
        try:
            return {"result": await sessions.call(sandbox_id, method, params)}
        except SessionError as error:
            return _error(error)

    if deployments is not None:

        @app.post("/deployments/{deployment_id}/start")
        async def start(deployment_id: str):
            try:
                return {"result": await deployments.start(deployment_id)}
            except SessionError as error:
                return _error(error)

        @app.post("/deployments/{deployment_id}/predict")
        async def predict(deployment_id: str, body: dict[str, Any] = Body(default={})):
            try:
                result = await deployments.predict(
                    deployment_id, body.get("files", []), body.get("options") or {}
                )
                return {"result": result}
            except SessionError as error:
                return _error(error)

        @app.post("/deployments/{deployment_id}/stop")
        async def stop(deployment_id: str):
            await deployments.stop(deployment_id)
            return {"result": {"stopped": True}}

    return app
