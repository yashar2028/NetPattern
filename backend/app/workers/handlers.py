"""Run one claimed job inside a pinned, isolated sandbox environment.

1. Resolve dataset paths (only the datasets folder is visible to sandboxes).
2. Find the capability packs the pipeline needs and add them (PLAN D23).
3. Build or reuse the environment (template + packs + engine snapshot).
4. Prefetch pretrained weights in a network-enabled step.
5. Run the job without network; stream every event into job_events.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.job import Job, JobStatus
from app.sandbox.environments import EnvironmentError_, EnvironmentManager
from app.sandbox.isolation import SandboxPolicy, sandbox_env, wrap
from app.sandbox.runner import run_engine_job, runtime_command, stderr_tail
from app.services import storage
from app.workers import queue

logger = logging.getLogger("netpattern.worker")

ENGINE_KINDS = ("smoke", "train", "sanity", "prefetch", "profile", "make_sample_dataset")
WEIGHT_KINDS = ("train", "sanity")
LOGGED_EVENTS = (
    "phase",
    "dataset_indexed",
    "model_built",
    "epoch_end",
    "evaluation",
    "run_failed",
    "warning",
)


class JobFailure(Exception):
    pass


class EventSink:
    """Persists engine events in order and logs the important ones."""

    def __init__(self, db: AsyncSession, job_id: str, start_seq: int) -> None:
        self.db, self.job_id, self.seq = db, job_id, start_seq

    async def __call__(self, event: dict[str, Any]) -> None:
        self.seq += 1
        event_type = str(event.get("type", "event"))[:64]
        payload = {key: value for key, value in event.items() if key != "type"}
        await queue.add_event(self.db, self.job_id, self.seq, event_type, payload)
        if event_type in LOGGED_EVENTS:
            logger.info("%s %s %s", self.job_id, event_type, json.dumps(payload, default=str)[:300])

    async def platform(self, event_type: str, **payload: Any) -> None:
        await self(dict(type=event_type, source="platform", **payload))


def _dataset_nodes(request: dict[str, Any]) -> list[dict[str, Any]]:
    nodes = request.get("pipeline", {}).get("nodes", [])
    return [
        n.setdefault("params", {})
        for n in nodes
        if isinstance(n, dict) and n.get("type") == "dataset"
    ]


def prepare_request(job: Job) -> tuple[dict[str, Any], list[Path]]:
    """The engine request with absolute dataset paths, plus extra writable paths."""
    request = copy.deepcopy(job.payload)
    request["run_id"], request["kind"] = job.id, job.kind
    targets = _dataset_nodes(request)
    if isinstance(request.get("dataset"), dict):
        targets.append(request["dataset"])
    for params in targets:
        if params.get("root"):
            params["root"] = str(storage.resolve_dataset_path(params["root"]))

    writable: list[Path] = []
    if job.kind == "make_sample_dataset":
        options = request.setdefault("options", {})
        out = storage.resolve_dataset_path(options.get("out") or "samples")
        out.mkdir(parents=True, exist_ok=True)
        options["out"] = str(out)
        writable.append(out)
    return request, writable


def needs_weights(request: dict[str, Any]) -> bool:
    for node in request.get("pipeline", {}).get("nodes", []):
        if node.get("type") != "model":
            continue
        arch = node.get("params", {}).get("architecture", {})
        if arch.get("kind") == "pretrained" and arch.get("base", {}).get("weights") is not None:
            return True
        if arch.get("kind") == "graph":
            return any(
                n.get("op") == "Backbone" and n.get("params", {}).get("pretrained", True)
                for n in arch.get("nodes", [])
            )
    return False


class JobHandler:
    def __init__(self, manager: EnvironmentManager | None = None) -> None:
        self.manager = manager or EnvironmentManager()

    async def required_packs(
        self, env_path: Path, run_dir: Path, request: dict[str, Any]
    ) -> list[str]:
        request_file = run_dir / "requirements-request.json"
        request_file.write_text(json.dumps(request), encoding="utf-8")
        policy = SandboxPolicy(
            read_only=[storage.datasets_root()], read_write=[run_dir], working_dir=run_dir
        )
        process = await asyncio.create_subprocess_exec(
            *wrap(
                runtime_command(env_path, "requirements", "--request", str(request_file)), policy
            ),
            cwd=str(run_dir),
            env=sandbox_env(env_path, run_dir, network=False),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=300)
        try:
            reply = json.loads(stdout.decode().strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError) as error:
            raise JobFailure(
                f"could not read pipeline requirements: {stderr.decode()[-500:]}"
            ) from error
        if process.returncode:
            raise JobFailure(reply.get("error") or "the pipeline spec is invalid")
        return list(reply.get("packs", []))

    async def run(
        self, db: AsyncSession, job: Job, should_cancel: Callable[[], bool]
    ) -> tuple[JobStatus, dict[str, Any] | None, str | None]:
        if job.kind not in ENGINE_KINDS:
            return JobStatus.FAILED, None, f"unknown job kind '{job.kind}'"
        run_dir = storage.run_dir(job.id)
        run_dir.mkdir(parents=True, exist_ok=True)
        job.run_dir = str(run_dir)
        await db.commit()
        events = EventSink(db, job.id, await queue.last_seq(db, job.id))

        try:
            request, writable = prepare_request(job)
        except ValueError as error:
            return JobStatus.FAILED, None, str(error)

        template = job.environment.get("template") or settings.DEFAULT_ENVIRONMENT_TEMPLATE
        packs = set(job.environment.get("packs", []))
        try:
            await events.platform("phase", name="preparing environment")
            if "pipeline" in request:
                base = await self.manager.ensure(db, template, sorted(packs))
                packs |= set(await self.required_packs(Path(base.store_path), run_dir, request))
            if job.kind == "make_sample_dataset" or "dataset" in request:
                packs.add("medical")
            environment = await self.manager.ensure(db, template, sorted(packs))
        except (EnvironmentError_, JobFailure) as error:
            return JobStatus.FAILED, None, str(error)

        job.environment_id = environment.id
        await db.commit()
        await events.platform(
            "environment_ready",
            environment_id=environment.id,
            env_hash=environment.env_hash,
            template=environment.template,
            packs=environment.packs,
            engine=environment.engine_version,
            isolation=settings.SANDBOX_ISOLATION,
        )
        env_path = Path(environment.store_path)
        cache = storage.cache_root()
        cache.mkdir(parents=True, exist_ok=True)

        if job.kind in WEIGHT_KINDS and needs_weights(request):
            await events.platform("phase", name="prefetching pretrained weights")
            prefetch = {**request, "kind": "prefetch"}
            outcome = await run_engine_job(
                env_path,
                run_dir,
                prefetch,
                events,
                should_cancel,
                SandboxPolicy(
                    read_only=[storage.datasets_root()], read_write=[cache], network=True
                ),
                result_name="prefetch-result.json",
            )
            if outcome.cancelled:
                return JobStatus.CANCELLED, None, "cancelled while downloading weights"
            if not outcome.result or outcome.result.get("status") != "completed":
                return (
                    JobStatus.FAILED,
                    outcome.result,
                    _failure(outcome.result, run_dir, "prefetch failed"),
                )

        outcome = await run_engine_job(
            env_path,
            run_dir,
            request,
            events,
            should_cancel,
            SandboxPolicy(
                read_only=[storage.datasets_root(), cache],
                read_write=writable,
                network=job.kind == "prefetch",
            ),
        )
        result = outcome.result
        if result is not None:
            result["environment_id"] = environment.id
            result["env_hash"] = environment.env_hash
        if outcome.cancelled or (result and result.get("status") == "cancelled"):
            return JobStatus.CANCELLED, result, None
        if outcome.timed_out:
            return JobStatus.FAILED, result, "the job exceeded its time limit"
        if outcome.exit_code == 0 and result and result.get("status") == "completed":
            return JobStatus.COMPLETED, result, None
        return (
            JobStatus.FAILED,
            result,
            _failure(result, run_dir, f"engine exited with code {outcome.exit_code}"),
        )


def _failure(result: dict[str, Any] | None, run_dir: Path, fallback: str) -> str:
    if result and result.get("error"):
        return str(result["error"])
    tail = stderr_tail(run_dir)
    return f"{fallback}\n{tail}" if tail else fallback
