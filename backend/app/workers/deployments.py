"""Deployments (PLAN §8.5): one model process per deployment, in the model's pinned environment.

- dedicated: started when the deployment is created (and when the worker starts), never idle-stopped
- serverless: started by the first predict call, stopped after DEPLOYMENT_IDLE_SECONDS

Input files of a predict call are written by the backend into the deployment's inference folder,
which the model process can read but not write; it has no network.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

from sqlalchemy import or_, select

from app.core.config import settings
from app.core.tiers import rate as tier_rate
from app.db.session import AsyncSessionLocal
from app.models.model import Deployment, DeploymentMode, DeploymentStatus, ModelVersion
from app.models.usage import UsageRecord
from app.sandbox.environments import EnvironmentError_, EnvironmentManager
from app.sandbox.isolation import SandboxPolicy
from app.sandbox.session import SessionClient, SessionError
from app.services import storage

logger = logging.getLogger("netpattern.worker")

LOAD_TIMEOUT_SECONDS = 300
PREDICT_TIMEOUT_SECONDS = 120


class DeploymentManager:
    def __init__(self, manager: EnvironmentManager) -> None:
        self.manager = manager
        self.sessions: dict[str, SessionClient] = {}
        self.modes: dict[str, str] = {}
        self.last_used: dict[str, float] = {}
        # Dedicated deployments are metered for the time they stay loaded (PLAN §11.2).
        self.owners: dict[str, tuple[str, str]] = {}  # id -> (owner_id, hardware tier)
        self.metered_at: dict[str, float] = {}
        self.locks: dict[str, asyncio.Lock] = {}

    def _lock(self, deployment_id: str) -> asyncio.Lock:
        return self.locks.setdefault(deployment_id, asyncio.Lock())

    async def start(self, deployment_id: str) -> dict[str, Any]:
        """(Re)start the model process and run the parity check."""
        async with self._lock(deployment_id):
            await self._stop(deployment_id)
            client = await self._launch(deployment_id)
            try:
                return await client.call("parity", {}, timeout=LOAD_TIMEOUT_SECONDS)
            except asyncio.TimeoutError as error:
                await self._stop(deployment_id)
                raise SessionError(-1, "the parity check did not finish in time") from error

    async def predict(
        self, deployment_id: str, files: list[str], options: dict[str, Any]
    ) -> dict[str, Any]:
        async with self._lock(deployment_id):
            client = self.sessions.get(deployment_id)
            if client is None or client.process is None or client.process.returncode is not None:
                client = await self._launch(deployment_id)  # serverless cold start
            self.last_used[deployment_id] = time.monotonic()
            try:
                return await client.call(
                    "predict", {"files": files, "options": options}, timeout=PREDICT_TIMEOUT_SECONDS
                )
            except (asyncio.TimeoutError, OSError) as error:
                await self._stop(deployment_id)
                raise SessionError(-1, f"the model did not answer: {error!r}") from error

    async def stop(self, deployment_id: str) -> None:
        async with self._lock(deployment_id):
            await self._stop(deployment_id)

    async def _launch(self, deployment_id: str) -> SessionClient:
        async with AsyncSessionLocal() as db:
            deployment = await db.get(Deployment, deployment_id)
            if deployment is None:
                raise SessionError(-1, "deployment not found")
            version = await db.get(ModelVersion, deployment.model_version_id)
            try:
                environment = await self.manager.get_ready(db, version.environment_id)
            except EnvironmentError_ as error:
                raise SessionError(-1, str(error)) from error
        model_dir = Path(version.path)
        inputs = storage.inference_root() / deployment_id
        inputs.mkdir(parents=True, exist_ok=True)
        client = SessionClient(
            Path(environment.store_path),
            storage.root() / "sessions" / deployment_id,
            SandboxPolicy(read_only=[model_dir, inputs, storage.cache_root()]),
        )
        device = "cpu" if deployment.hardware_tier.startswith("cpu") else None
        try:
            await client.start()
            await client.call(
                "load_model",
                {"model_dir": str(model_dir), "device": device},
                timeout=LOAD_TIMEOUT_SECONDS,
            )
        except SessionError:
            await client.close()
            raise
        except asyncio.TimeoutError as error:
            await client.close()
            raise SessionError(-1, "the model did not load in time") from error
        self.sessions[deployment_id] = client
        self.modes[deployment_id] = deployment.mode
        self.last_used[deployment_id] = time.monotonic()
        self.owners[deployment_id] = (deployment.owner_id, deployment.hardware_tier)
        self.metered_at[deployment_id] = time.monotonic()
        logger.info("Started %s deployment %s", deployment.mode, deployment_id)
        return client

    async def _stop(self, deployment_id: str) -> None:
        await self._meter_one(deployment_id)
        client = self.sessions.pop(deployment_id, None)
        self.modes.pop(deployment_id, None)
        self.last_used.pop(deployment_id, None)
        self.owners.pop(deployment_id, None)
        self.metered_at.pop(deployment_id, None)
        if client is not None:
            await client.close()

    async def _meter_one(self, deployment_id: str) -> None:
        if self.modes.get(deployment_id) != DeploymentMode.DEDICATED:
            return  # serverless calls are metered per request (inference_records)
        now = time.monotonic()
        seconds = round(now - self.metered_at.get(deployment_id, now), 3)
        if seconds <= 0:
            return
        owner_id, tier = self.owners[deployment_id]
        self.metered_at[deployment_id] = now
        async with AsyncSessionLocal() as db:
            db.add(
                UsageRecord(
                    owner_id=owner_id,
                    deployment_id=deployment_id,
                    hardware_tier=tier,
                    seconds=seconds,
                    units=round(seconds * tier_rate(tier), 3),
                )
            )
            await db.commit()

    async def meter(self) -> None:
        """Write the loaded time of dedicated deployments to the ledger (called every minute)."""
        for deployment_id in list(self.sessions):
            await self._meter_one(deployment_id)

    async def stop_idle(self) -> None:
        cutoff = time.monotonic() - settings.DEPLOYMENT_IDLE_SECONDS
        idle = [
            d
            for d, used in self.last_used.items()
            if used < cutoff and self.modes.get(d) == DeploymentMode.SERVERLESS
        ]
        for deployment_id in idle:
            await self.stop(deployment_id)
            logger.info("Stopped idle serverless deployment %s", deployment_id)

    async def start_dedicated(self) -> None:
        """Bring dedicated deployments back after a worker restart."""
        async with AsyncSessionLocal() as db:
            rows = (
                await db.execute(
                    select(Deployment.id).where(
                        Deployment.mode == DeploymentMode.DEDICATED,
                        Deployment.status == DeploymentStatus.ACTIVE,
                        # Only the ones assigned to this worker (or to none, before Phase 8).
                        or_(
                            Deployment.worker_id == settings.WORKER_ID,
                            Deployment.worker_id.is_(None),
                        ),
                    )
                )
            ).scalars()
            ids = list(rows)
        for deployment_id in ids:
            try:
                async with self._lock(deployment_id):
                    await self._launch(deployment_id)
            except SessionError as error:
                logger.warning("Could not restart deployment %s: %s", deployment_id, error)

    async def stop_all(self) -> None:
        for deployment_id in list(self.sessions):
            await self._stop(deployment_id)
