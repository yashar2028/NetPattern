"""Sandboxes and their pipelines (immutable versions of the canvas spec)."""

from fastapi import APIRouter, Depends, Query, Response, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, not_found
from app.core.errors import ApiError
from app.core.pagination import DEFAULT_LIMIT, Page, paginate
from app.db.session import get_db
from app.models.job import Job, JobStatus
from app.models.sandbox import Pipeline, PipelineVersion, Sandbox
from app.models.user import User
from app.sandbox.environments import KNOWN_PACKS, KNOWN_TEMPLATES
from app.schemas.api import (
    PipelineCreate,
    PipelineOut,
    PipelineVersionCreate,
    PipelineVersionOut,
    SandboxCreate,
    SandboxOut,
    SandboxUpdate,
)

router = APIRouter(prefix="/v1", tags=["sandboxes"])


async def owned_sandbox(db: AsyncSession, sandbox_id: str, user: User) -> Sandbox:
    sandbox = await db.get(Sandbox, sandbox_id)
    if sandbox is None or sandbox.owner_id != user.id:
        raise not_found("Sandbox")
    return sandbox


async def owned_pipeline(db: AsyncSession, pipeline_id: str, user: User) -> Pipeline:
    pipeline = await db.get(Pipeline, pipeline_id)
    if pipeline is None:
        raise not_found("Pipeline")
    await owned_sandbox(db, pipeline.sandbox_id, user)
    return pipeline


@router.get("/environment-templates")
async def environment_templates() -> dict:
    return {"templates": list(KNOWN_TEMPLATES), "packs": list(KNOWN_PACKS), "tiers": ["cpu", "gpu"]}


# --------------------------------------------------------------------------- sandboxes


@router.post("/sandboxes", response_model=SandboxOut, status_code=status.HTTP_201_CREATED)
async def create_sandbox(
    body: SandboxCreate, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
) -> Sandbox:
    sandbox = Sandbox(owner_id=user.id, name=body.name, template=body.template)
    db.add(sandbox)
    await db.commit()
    await db.refresh(sandbox)
    return sandbox


@router.get("/sandboxes", response_model=Page[SandboxOut])
async def list_sandboxes(
    limit: int = Query(DEFAULT_LIMIT),
    cursor: str | None = None,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    rows, next_cursor = await paginate(
        db, select(Sandbox).where(Sandbox.owner_id == user.id), Sandbox, limit, cursor
    )
    return {"data": rows, "next_cursor": next_cursor}


@router.get("/sandboxes/{sandbox_id}", response_model=SandboxOut)
async def get_sandbox(
    sandbox_id: str, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
) -> Sandbox:
    return await owned_sandbox(db, sandbox_id, user)


@router.patch("/sandboxes/{sandbox_id}", response_model=SandboxOut)
async def rename_sandbox(
    sandbox_id: str,
    body: SandboxUpdate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Sandbox:
    sandbox = await owned_sandbox(db, sandbox_id, user)
    sandbox.name = body.name
    await db.commit()
    return sandbox


@router.delete("/sandboxes/{sandbox_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_sandbox(
    sandbox_id: str, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
) -> Response:
    sandbox = await owned_sandbox(db, sandbox_id, user)
    active = await db.scalar(
        select(func.count(Job.id)).where(
            Job.sandbox_id == sandbox.id, Job.status.in_([JobStatus.QUEUED, JobStatus.RUNNING])
        )
    )
    if active:
        raise ApiError(status.HTTP_409_CONFLICT, "runs_active", "Cancel the sandbox's runs first")
    await db.delete(sandbox)
    await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/sandboxes/{sandbox_id}/environment/upgrade", response_model=SandboxOut)
async def upgrade_environment(
    sandbox_id: str, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
) -> Sandbox:
    """Unpin the engine: the next run uses (and pins) the newest engine version."""
    sandbox = await owned_sandbox(db, sandbox_id, user)
    sandbox.engine_snapshot = sandbox.engine_version = None
    await db.commit()
    return sandbox


# --------------------------------------------------------------------------- pipelines


async def _add_version(db: AsyncSession, pipeline: Pipeline, spec: dict) -> PipelineVersion:
    number = (
        await db.scalar(
            select(func.max(PipelineVersion.number)).where(
                PipelineVersion.pipeline_id == pipeline.id
            )
        )
        or 0
    ) + 1
    version = PipelineVersion(pipeline_id=pipeline.id, number=number, spec=spec)
    db.add(version)
    return version


async def _pipeline_out(db: AsyncSession, pipeline: Pipeline) -> PipelineOut:
    latest = await db.scalar(
        select(func.max(PipelineVersion.number)).where(PipelineVersion.pipeline_id == pipeline.id)
    )
    return PipelineOut.model_validate(pipeline).model_copy(update={"latest_version": latest})


@router.post(
    "/sandboxes/{sandbox_id}/pipelines",
    response_model=PipelineOut,
    status_code=status.HTTP_201_CREATED,
    tags=["pipelines"],
)
async def create_pipeline(
    sandbox_id: str,
    body: PipelineCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> PipelineOut:
    sandbox = await owned_sandbox(db, sandbox_id, user)
    spec = PipelineVersionCreate(spec=body.spec).spec if body.spec is not None else None
    pipeline = Pipeline(sandbox_id=sandbox.id, name=body.name)
    db.add(pipeline)
    await db.flush()
    if spec is not None:
        await _add_version(db, pipeline, spec)
    await db.commit()
    await db.refresh(pipeline)
    return await _pipeline_out(db, pipeline)


@router.get(
    "/sandboxes/{sandbox_id}/pipelines", response_model=Page[PipelineOut], tags=["pipelines"]
)
async def list_pipelines(
    sandbox_id: str,
    limit: int = Query(DEFAULT_LIMIT),
    cursor: str | None = None,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    await owned_sandbox(db, sandbox_id, user)
    rows, next_cursor = await paginate(
        db, select(Pipeline).where(Pipeline.sandbox_id == sandbox_id), Pipeline, limit, cursor
    )
    return {"data": [await _pipeline_out(db, p) for p in rows], "next_cursor": next_cursor}


@router.get("/pipelines/{pipeline_id}", response_model=PipelineOut, tags=["pipelines"])
async def get_pipeline(
    pipeline_id: str, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
) -> PipelineOut:
    return await _pipeline_out(db, await owned_pipeline(db, pipeline_id, user))


@router.delete(
    "/pipelines/{pipeline_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["pipelines"]
)
async def delete_pipeline(
    pipeline_id: str, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
) -> Response:
    pipeline = await owned_pipeline(db, pipeline_id, user)
    await db.delete(pipeline)
    await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/pipelines/{pipeline_id}/versions",
    response_model=PipelineVersionOut,
    status_code=status.HTTP_201_CREATED,
    tags=["pipelines"],
)
async def save_version(
    pipeline_id: str,
    body: PipelineVersionCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> PipelineVersion:
    pipeline = await owned_pipeline(db, pipeline_id, user)
    version = await _add_version(db, pipeline, body.spec)
    await db.commit()
    await db.refresh(version)
    return version


@router.get(
    "/pipelines/{pipeline_id}/versions", response_model=list[PipelineVersionOut], tags=["pipelines"]
)
async def list_versions(
    pipeline_id: str, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
) -> list[PipelineVersionOut]:
    await owned_pipeline(db, pipeline_id, user)
    rows = (
        await db.execute(
            select(PipelineVersion)
            .where(PipelineVersion.pipeline_id == pipeline_id)
            .order_by(PipelineVersion.number.desc())
        )
    ).scalars()
    return [PipelineVersionOut.model_validate(v).model_copy(update={"spec": None}) for v in rows]


@router.get(
    "/pipelines/{pipeline_id}/versions/{number}",
    response_model=PipelineVersionOut,
    tags=["pipelines"],
)
async def get_version(
    pipeline_id: str,
    number: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> PipelineVersion:
    await owned_pipeline(db, pipeline_id, user)
    version = await db.scalar(
        select(PipelineVersion).where(
            PipelineVersion.pipeline_id == pipeline_id, PipelineVersion.number == number
        )
    )
    if version is None:
        raise not_found("Pipeline version")
    return version
