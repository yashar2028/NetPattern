"""Model registry (PLAN §8.3): completed training runs frozen as numbered model versions."""

import asyncio
import shutil
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, not_found
from app.api.runs import owned_run
from app.core.errors import ApiError
from app.db.session import get_db
from app.models.job import Environment, Job, JobStatus
from app.models.model import Model, ModelVersion
from app.models.user import User
from app.schemas.api import ModelCreate, ModelOut, ModelVersionOut, RunOut
from app.services import storage
from app.workers import queue

router = APIRouter(prefix="/v1", tags=["models"])

# What a model version keeps from its run directory.
MODEL_FILES = ("model.json", "parity.json")
# The first engine with the export job kind.
EXPORT_MIN_ENGINE = (0, 5, 0)


def _invalid(detail: str) -> ApiError:
    return ApiError(
        status.HTTP_422_UNPROCESSABLE_CONTENT, "invalid_model", "Cannot register the run", detail
    )


def _copy_model_files(run_dir: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(run_dir / "checkpoints" / "best.pt", target / "best.pt")
    for name in MODEL_FILES:
        if (run_dir / name).exists():
            shutil.copyfile(run_dir / name, target / name)
    if (run_dir / "parity").is_dir():
        shutil.copytree(run_dir / "parity", target / "parity", dirs_exist_ok=True)


async def owned_model(db: AsyncSession, model_id: str, user: User) -> Model:
    model = await db.get(Model, model_id)
    if model is None or model.owner_id != user.id:
        raise not_found("Model")
    return model


async def owned_version(db: AsyncSession, version_id: str, user: User) -> ModelVersion:
    version = await db.get(ModelVersion, version_id)
    if version is None:
        raise not_found("Model version")
    await owned_model(db, version.model_id, user)
    return version


async def _model_out(db: AsyncSession, model: Model) -> ModelOut:
    versions = (
        await db.execute(
            select(ModelVersion)
            .where(ModelVersion.model_id == model.id)
            .order_by(ModelVersion.number)
        )
    ).scalars()
    return ModelOut.model_validate(model).model_copy(
        update={"versions": [ModelVersionOut.model_validate(v) for v in versions]}
    )


@router.post("/models", response_model=ModelOut, status_code=status.HTTP_201_CREATED)
async def register_model(
    body: ModelCreate, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
) -> ModelOut:
    run = await owned_run(db, body.run_id, user)
    if run.kind != "train" or run.status != JobStatus.COMPLETED:
        raise _invalid("only completed training runs can become models")
    run_dir = Path(run.run_dir or "")
    if not (run_dir / "model.json").exists() or not (run_dir / "checkpoints/best.pt").exists():
        raise _invalid("the run has no model description; re-run it with engine 0.3.0 or newer")
    if not run.environment_id:
        raise _invalid("the run has no pinned environment")

    if body.model_id:
        model = await owned_model(db, body.model_id, user)
    elif body.name:
        model = Model(owner_id=user.id, sandbox_id=run.sandbox_id, name=body.name)
        db.add(model)
        await db.flush()
    else:
        raise _invalid("give a name for a new model or the model_id to add a version to")

    number = (
        await db.scalar(
            select(func.max(ModelVersion.number)).where(ModelVersion.model_id == model.id)
        )
        or 0
    ) + 1
    result = run.result or {}
    evaluation = result.get("evaluation") or {}
    version = ModelVersion(
        model_id=model.id,
        number=number,
        run_id=run.id,
        environment_id=run.environment_id,
        path="",
        task=result.get("task") or {},
        input=(result.get("model") or {}).get("input") or {},
        metrics={split: value.get("metrics", {}) for split, value in evaluation.items()},
    )
    db.add(version)
    await db.flush()
    target = storage.models_root() / version.id
    await asyncio.to_thread(_copy_model_files, run_dir, target)
    version.path = str(target)
    await db.commit()
    return await _model_out(db, model)


@router.get("/models", response_model=list[ModelOut])
async def list_models(
    user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
) -> list[ModelOut]:
    models = (
        await db.execute(
            select(Model).where(Model.owner_id == user.id).order_by(Model.created_at.desc())
        )
    ).scalars()
    return [await _model_out(db, model) for model in models]


@router.get("/models/{model_id}", response_model=ModelOut)
async def get_model(
    model_id: str, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
) -> ModelOut:
    return await _model_out(db, await owned_model(db, model_id, user))


class ExportRequest(BaseModel):
    formats: list[Literal["state_dict", "torchscript", "onnx"]] = [
        "state_dict",
        "torchscript",
        "onnx",
    ]


@router.post(
    "/models/{model_id}/versions/{number}/export",
    response_model=RunOut,
    status_code=status.HTTP_202_ACCEPTED,
)
async def export_version(
    model_id: str,
    number: int,
    body: ExportRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Job:
    """Export runs as a job in the model's own environment; download the files from the run."""
    model = await owned_model(db, model_id, user)
    version = await db.scalar(
        select(ModelVersion).where(ModelVersion.model_id == model.id, ModelVersion.number == number)
    )
    if version is None:
        raise not_found("Model version")
    # The model's template and packs (plus ONNX if asked). Its own engine when that engine can
    # export; otherwise the current one: export only converts the trained weights.
    environment = await db.get(Environment, version.environment_id)
    target: dict = {"template": environment.template, "packs": list(environment.packs)}
    if _engine(environment.engine_version) >= EXPORT_MIN_ENGINE:
        target["engine_snapshot"] = environment.engine_snapshot
    if "onnx" in body.formats:
        target["packs"] = sorted({*target["packs"], "export"})
    return await queue.enqueue(
        db,
        "export",
        {"model_dir": version.path, "options": {"formats": body.formats}},
        target,
        owner_id=user.id,
        sandbox_id=model.sandbox_id,
    )


def _engine(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split(".")[:3] if part.isdigit())
