"""Sweeps (PLAN Phase 8): one pipeline version, run many times with chosen values changed.

A parameter path names a node type and a field inside its params, e.g. `trainer.optimizer.lr`,
`trainer.batch_size`, `dataset.split.fold` (k-fold cross-validation) or `task.threshold`.
Each combination becomes an ordinary training run; compare them on the comparison page.
"""

import copy
import itertools
import random
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, status
from pydantic import AfterValidator, BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, not_found
from app.api.sandboxes import owned_sandbox
from app.core.errors import ApiError
from app.db.session import get_db
from app.models.job import Job
from app.models.sandbox import Pipeline, PipelineVersion
from app.models.sweep import Sweep
from app.models.user import User
from app.schemas.api import RunOut, _known_tier
from app.services.runs import RunRequestError, resolve_datasets
from app.workers import queue

router = APIRouter(prefix="/v1", tags=["sweeps"])

MAX_SWEEP_RUNS = 32
PATH = r"^(task|dataset|transforms|model|trainer|evaluator)(\.[A-Za-z0-9_]+)+$"


class SweepParameter(BaseModel):
    path: str = Field(pattern=PATH, examples=["trainer.optimizer.lr"])
    values: list[Any] = Field(min_length=1, max_length=20)


class SweepCreate(BaseModel):
    pipeline_version_id: str
    parameters: list[SweepParameter] = Field(min_length=1, max_length=4)
    mode: Literal["grid", "random"] = "grid"
    max_runs: int = Field(default=16, ge=1, le=MAX_SWEEP_RUNS)
    hardware_tier: Annotated[str, AfterValidator(_known_tier)] = "cpu"


class SweepOut(BaseModel):
    id: str
    sandbox_id: str
    pipeline_version_id: str
    parameters: list[dict[str, Any]]
    mode: str
    runs: list[RunOut]


def _invalid(detail: str) -> ApiError:
    return ApiError(
        status.HTTP_422_UNPROCESSABLE_CONTENT, "invalid_sweep", "Cannot start the sweep", detail
    )


def combinations(parameters: list[SweepParameter], mode: str, max_runs: int) -> list[dict]:
    grid = [
        dict(zip([p.path for p in parameters], values, strict=True))
        for values in itertools.product(*[p.values for p in parameters])
    ]
    if mode == "random":
        random.Random(0).shuffle(grid)
    elif len(grid) > max_runs:
        raise _invalid(f"the grid has {len(grid)} runs, more than max_runs ({max_runs})")
    return grid[:max_runs]


def apply(spec: dict[str, Any], values: dict[str, Any]) -> dict[str, Any]:
    """The spec with each path set to its value (nested dicts are created as needed)."""
    spec = copy.deepcopy(spec)
    for path, value in values.items():
        kind, *keys = path.split(".")
        if kind == "task":
            target = spec.setdefault("task", {})
        else:
            node = next((n for n in spec.get("nodes", []) if n.get("type") == kind), None)
            if node is None:
                raise _invalid(f"the pipeline has no {kind} node (path {path})")
            target = node.setdefault("params", {})
        for key in keys[:-1]:
            target = target.setdefault(key, {})
            if not isinstance(target, dict):
                raise _invalid(f"{path} goes through a value that is not an object")
        target[keys[-1]] = value
    return spec


async def _out(db: AsyncSession, sweep: Sweep) -> SweepOut:
    jobs = {
        job.id: job
        for job in (await db.execute(select(Job).where(Job.id.in_(sweep.run_ids)))).scalars()
    }
    runs = [
        RunOut.model_validate(jobs[i]).model_copy(update={"result": None})
        for i in sweep.run_ids
        if i in jobs
    ]
    return SweepOut(
        id=sweep.id,
        sandbox_id=sweep.sandbox_id,
        pipeline_version_id=sweep.pipeline_version_id,
        parameters=sweep.parameters,
        mode=sweep.mode,
        runs=runs,
    )


@router.post(
    "/sandboxes/{sandbox_id}/sweeps", response_model=SweepOut, status_code=status.HTTP_201_CREATED
)
async def start_sweep(
    sandbox_id: str,
    body: SweepCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> SweepOut:
    sandbox = await owned_sandbox(db, sandbox_id, user)
    version = await db.get(PipelineVersion, body.pipeline_version_id)
    pipeline = await db.get(Pipeline, version.pipeline_id) if version else None
    if version is None or pipeline is None or pipeline.sandbox_id != sandbox.id:
        raise not_found("Pipeline version")
    combos = combinations(body.parameters, body.mode, body.max_runs)
    sweep = Sweep(
        owner_id=user.id,
        sandbox_id=sandbox.id,
        pipeline_version_id=version.id,
        parameters=[p.model_dump() for p in body.parameters],
        mode=body.mode,
        run_ids=[],
    )
    db.add(sweep)
    await db.flush()
    payloads = []
    for values in combos:
        try:
            spec = await resolve_datasets(db, apply(version.spec, values), user)
        except RunRequestError as error:
            raise _invalid(str(error)) from error
        payloads.append({"pipeline": spec, "sweep": {"id": sweep.id, "values": values}})
    await db.commit()

    environment = {"template": sandbox.template}
    if sandbox.engine_snapshot:
        environment["engine_snapshot"] = sandbox.engine_snapshot
    run_ids = []
    for payload in payloads:
        job = await queue.enqueue(
            db,
            "train",
            payload,
            environment,
            hardware_tier=body.hardware_tier,
            owner_id=user.id,
            sandbox_id=sandbox.id,
            pipeline_version_id=version.id,
        )
        run_ids.append(job.id)
    sweep = await db.get(Sweep, sweep.id)
    sweep.run_ids = run_ids
    await db.commit()
    return await _out(db, sweep)


@router.get("/sweeps/{sweep_id}", response_model=SweepOut)
async def get_sweep(
    sweep_id: str, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
) -> SweepOut:
    sweep = await db.get(Sweep, sweep_id)
    if sweep is None:
        raise not_found("Sweep")
    await owned_sandbox(db, sweep.sandbox_id, user)
    return await _out(db, sweep)


@router.get("/sandboxes/{sandbox_id}/sweeps", response_model=list[SweepOut])
async def list_sweeps(
    sandbox_id: str, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
) -> list[SweepOut]:
    await owned_sandbox(db, sandbox_id, user)
    rows = await db.execute(
        select(Sweep).where(Sweep.sandbox_id == sandbox_id).order_by(Sweep.created_at.desc())
    )
    return [await _out(db, sweep) for sweep in rows.scalars()]
