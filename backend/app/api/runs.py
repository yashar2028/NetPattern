"""Runs (jobs started through the API), their live events, checkpoints, predictions, usage."""

import asyncio
import json
import re
import shutil
from pathlib import PurePosixPath
from typing import Literal
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, Query, Request, UploadFile, status
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, not_found
from app.api.sandboxes import owned_sandbox
from app.core.config import settings
from app.core.errors import ApiError
from app.core.pagination import DEFAULT_LIMIT, Page, paginate
from app.db.session import AsyncSessionLocal, get_db
from app.models.job import FINISHED_STATUSES, Job, JobStatus
from app.models.sandbox import Pipeline, PipelineVersion
from app.models.usage import UsageRecord
from app.models.user import User
from app.schemas.api import RunCreate, RunEventOut, RunOut, UsageOut
from app.services import storage
from app.services.runs import (
    RunRequestError,
    checkpoint_path,
    dataset_folder,
    missing_inputs,
    owned_dataset_version,
    resolve_datasets,
)
from app.workers import queue

router = APIRouter(prefix="/v1", tags=["runs"])

STREAM_POLL_SECONDS = 1.0
STREAM_KEEPALIVE_SECONDS = 15.0
CHUNK = 1024 * 1024
# Must match what the engine can read (netpattern_engine.data.readers).
IMAGE_SUFFIXES = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
    ".dcm",
    ".dicom",
    ".nii",
    ".nii.gz",
)
UNSAFE_CHARACTERS = re.compile(r"[^A-Za-z0-9._-]+")
MASK_FILE = re.compile(r"^\d{5}\.png$")


async def owned_run(db: AsyncSession, run_id: str, user: User) -> Job:
    job = await db.get(Job, run_id)
    if job is None or job.owner_id != user.id:
        raise not_found("Run")
    return job


@router.post(
    "/sandboxes/{sandbox_id}/runs", response_model=RunOut, status_code=status.HTTP_202_ACCEPTED
)
async def start_run(
    sandbox_id: str,
    body: RunCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Job:
    sandbox = await owned_sandbox(db, sandbox_id, user)
    version = await db.get(PipelineVersion, body.pipeline_version_id)
    pipeline = await db.get(Pipeline, version.pipeline_id) if version else None
    if version is None or pipeline is None or pipeline.sandbox_id != sandbox.id:
        raise not_found("Pipeline version")
    try:
        spec, version_ids = await resolve_datasets(db, version.spec, user)
    except RunRequestError as error:
        raise ApiError(
            status.HTTP_422_UNPROCESSABLE_CONTENT, "invalid_run", "Cannot start the run", str(error)
        ) from error

    environment = {"template": sandbox.template}
    if sandbox.engine_snapshot:
        environment["engine_snapshot"] = sandbox.engine_snapshot
    return await queue.enqueue(
        db,
        body.kind,
        {"pipeline": spec},
        environment,
        hardware_tier=body.hardware_tier,
        owner_id=user.id,
        sandbox_id=sandbox.id,
        pipeline_version_id=version.id,
        dataset_version_ids=version_ids,
    )


@router.get("/runs", response_model=Page[RunOut])
async def list_runs(
    sandbox_id: str | None = None,
    status_filter: JobStatus | None = Query(None, alias="status"),
    limit: int = Query(DEFAULT_LIMIT),
    cursor: str | None = None,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    query = select(Job).where(Job.owner_id == user.id)
    if sandbox_id:
        query = query.where(Job.sandbox_id == sandbox_id)
    if status_filter:
        query = query.where(Job.status == status_filter)
    rows, next_cursor = await paginate(db, query, Job, limit, cursor)
    # Lists stay small: full results are on GET /v1/runs/{id}.
    return {
        "data": [RunOut.model_validate(r).model_copy(update={"result": None}) for r in rows],
        "next_cursor": next_cursor,
    }


@router.get("/runs/{run_id}", response_model=RunOut)
async def get_run(
    run_id: str, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
) -> Job:
    return await owned_run(db, run_id, user)


@router.post("/runs/{run_id}/cancel", response_model=RunOut)
async def cancel_run(
    run_id: str, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
) -> Job:
    job = await owned_run(db, run_id, user)
    await queue.request_cancel(db, job.id)
    return await db.get(Job, job.id, populate_existing=True)


@router.post("/runs/{run_id}/rerun", response_model=RunOut, status_code=status.HTTP_202_ACCEPTED)
async def rerun(
    run_id: str, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
) -> Job:
    """Run again with the same request and, when it exists, the exact same environment."""
    job = await owned_run(db, run_id, user)
    missing = await missing_inputs(db, job)
    if missing:
        raise ApiError(
            status.HTTP_409_CONFLICT,
            "inputs_missing",
            "Cannot rerun",
            "no longer available: " + "; ".join(missing),
        )
    environment = (
        {"environment_id": job.environment_id} if job.environment_id else dict(job.environment)
    )
    return await queue.enqueue(
        db,
        job.kind,
        job.payload,
        environment,
        hardware_tier=job.hardware_tier,
        owner_id=user.id,
        sandbox_id=job.sandbox_id,
        pipeline_version_id=job.pipeline_version_id,
        dataset_version_ids=job.dataset_version_ids,
    )


# --------------------------------------------------------------------------- checkpoints


@router.get(
    "/runs/{run_id}/checkpoints/{name}",
    response_class=FileResponse,
    responses={200: {"content": {"application/octet-stream": {}}}},
)
async def download_checkpoint(
    run_id: str,
    name: Literal["best", "last"],
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> FileResponse:
    """The run's checkpoint: weights plus the task state and input spec needed to use them."""
    job = await owned_run(db, run_id, user)
    try:
        path = checkpoint_path(job.id, name)
    except ValueError:
        raise not_found("Checkpoint") from None
    if not path.is_file():
        raise not_found("Checkpoint")
    return FileResponse(
        path, media_type="application/octet-stream", filename=f"{job.id}-{name}.pt"
    )


# --------------------------------------------------------------------------- predictions


def _input_name(index: int, filename: str | None) -> str:
    """A safe file name that keeps the original name (for the results) and its extension."""
    name = PurePosixPath((filename or "").replace("\\", "/")).name
    if not name.lower().endswith(IMAGE_SUFFIXES):
        raise ApiError(
            status.HTTP_400_BAD_REQUEST,
            "unsupported_file",
            "Unsupported file",
            f"'{name}' is not a supported image ({', '.join(IMAGE_SUFFIXES)})",
        )
    return f"{index:04d}_{UNSAFE_CHARACTERS.sub('_', name)[-100:]}"


async def _save_inputs(files: list[UploadFile], user: User) -> str:
    """Store uploaded images; returns their folder relative to the prediction inputs root."""
    if len(files) > settings.PREDICT_MAX_FILES:
        raise ApiError(
            status.HTTP_413_CONTENT_TOO_LARGE,
            "too_many_files",
            "Too many files",
            f"send at most {settings.PREDICT_MAX_FILES} images, or use a dataset version",
        )
    relative = f"{user.id}/{uuid4().hex}"
    folder = storage.predict_inputs_root() / relative
    folder.mkdir(parents=True)
    try:
        total = 0
        for index, upload in enumerate(files):
            with (folder / _input_name(index, upload.filename)).open("wb") as out:
                while chunk := await upload.read(CHUNK):
                    total += len(chunk)
                    if total > settings.PREDICT_MAX_UPLOAD_BYTES:
                        raise ApiError(
                            status.HTTP_413_CONTENT_TOO_LARGE,
                            "upload_too_large",
                            "Upload too large",
                            f"the limit is {settings.PREDICT_MAX_UPLOAD_BYTES} bytes",
                        )
                    out.write(chunk)
    except BaseException:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    return relative


@router.post(
    "/runs/{run_id}/predict",
    response_model=RunOut,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["predictions"],
)
async def predict(
    run_id: str,
    files: list[UploadFile] | None = File(None, description="images to predict on"),
    dataset_version_id: str | None = Form(None, description="or: every image of this version"),
    subpath: str | None = Form(None, description="folder inside the dataset version"),
    checkpoint: Literal["best", "last"] = Form("best"),
    hardware_tier: Literal["cpu", "gpu"] = Form("cpu"),
    score_threshold: float = Form(0.5, ge=0, le=1, description="detection: minimum box score"),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Job:
    """Predict with a trained model, in the exact environment it was trained in.

    The result (`GET /v1/runs/{id}`) lists one prediction per image (per slice for
    volumes). Segmentation masks are PNG files: `GET /v1/runs/{id}/predictions/{file}`.
    """
    source = await owned_run(db, run_id, user)
    if source.kind != "train" or source.status != JobStatus.COMPLETED:
        raise ApiError(
            status.HTTP_409_CONFLICT,
            "run_not_predictable",
            "Not a completed training run",
            "predictions need a completed training run",
        )
    if not source.environment_id or "pipeline" not in (source.payload or {}):
        raise ApiError(
            status.HTTP_409_CONFLICT,
            "run_not_predictable",
            "Not a completed training run",
            "the run has no recorded environment",
        )
    if not checkpoint_path(source.id, checkpoint).is_file():
        raise not_found("Checkpoint")
    if bool(files) == bool(dataset_version_id):
        raise ApiError(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "invalid_predict",
            "Invalid prediction request",
            "send either image files or a dataset_version_id",
        )

    version_ids: list[str] = []
    if dataset_version_id:
        try:
            version = await owned_dataset_version(db, dataset_version_id, user)
            inputs = {"source": "dataset", "root": dataset_folder(version, subpath)}
        except RunRequestError as error:
            raise ApiError(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                "invalid_predict",
                "Invalid prediction request",
                str(error),
            ) from error
        version_ids.append(version.id)
    else:
        inputs = {"source": "upload", "root": await _save_inputs(files or [], user)}

    payload = {
        "pipeline": source.payload["pipeline"],
        "source_run_id": source.id,
        "checkpoint": checkpoint,
        "inputs": inputs,
        "options": {"score_threshold": score_threshold},
    }
    return await queue.enqueue(
        db,
        "predict",
        payload,
        {"environment_id": source.environment_id},
        hardware_tier=hardware_tier,
        owner_id=user.id,
        sandbox_id=source.sandbox_id,
        pipeline_version_id=source.pipeline_version_id,
        dataset_version_ids=version_ids,
    )


@router.get(
    "/runs/{run_id}/predictions/{name}",
    response_class=FileResponse,
    responses={200: {"content": {"image/png": {}}}},
    tags=["predictions"],
)
async def prediction_file(
    run_id: str,
    name: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> FileResponse:
    """A segmentation mask written by a prediction run (pixel value = class index)."""
    job = await owned_run(db, run_id, user)
    path = storage.run_dir(job.id) / "predictions" / name
    if not MASK_FILE.match(name) or not path.is_file():
        raise not_found("Prediction file")
    return FileResponse(path, media_type="image/png")


@router.get("/runs/{run_id}/events", response_model=list[RunEventOut])
async def list_events(
    run_id: str,
    after: int = Query(0, ge=0),
    limit: int = Query(500, ge=1, le=2000),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    job = await owned_run(db, run_id, user)
    return await queue.events_after(db, job.id, after, limit)


@router.get("/runs/{run_id}/events/stream")
async def stream_events(
    run_id: str,
    request: Request,
    after: int = Query(0, ge=0),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """Server-Sent Events; reconnect with Last-Event-ID to resume (PLAN §8.7)."""
    job = await owned_run(db, run_id, user)
    start = int(request.headers.get("last-event-id") or after)

    async def events():
        seq, idle = start, 0.0
        while not await request.is_disconnected():
            async with AsyncSessionLocal() as session:
                batch = await queue.events_after(session, job.id, seq)
                current = await session.get(Job, job.id)
            for event in batch:
                seq = event.seq
                data = json.dumps(event.payload, default=str)
                yield f"id: {event.seq}\nevent: {event.type}\ndata: {data}\n\n"
            if current is None or (current.status in FINISHED_STATUSES and not batch):
                status_value = current.status.value if current else "deleted"
                yield f"event: end\ndata: {json.dumps({'status': status_value})}\n\n"
                return
            idle = 0.0 if batch else idle + STREAM_POLL_SECONDS
            if idle >= STREAM_KEEPALIVE_SECONDS:
                idle = 0.0
                yield ": keep-alive\n\n"
            await asyncio.sleep(STREAM_POLL_SECONDS)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/usage", response_model=UsageOut, tags=["usage"])
async def usage(
    user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
) -> UsageOut:
    """Compute time used, by hardware tier and sandbox (no prices yet)."""
    mine = UsageRecord.owner_id == user.id
    by_tier = dict(
        (
            await db.execute(
                select(UsageRecord.hardware_tier, func.sum(UsageRecord.seconds))
                .where(mine)
                .group_by(UsageRecord.hardware_tier)
            )
        ).all()
    )
    by_sandbox = dict(
        (
            await db.execute(
                select(UsageRecord.sandbox_id, func.sum(UsageRecord.seconds))
                .where(mine)
                .group_by(UsageRecord.sandbox_id)
            )
        ).all()
    )
    jobs = await db.scalar(select(func.count(UsageRecord.id)).where(mine)) or 0
    return UsageOut(
        total_seconds=round(sum(by_tier.values()), 2),
        jobs=jobs,
        by_tier={tier: round(seconds, 2) for tier, seconds in by_tier.items()},
        by_sandbox={
            str(sandbox or "none"): round(seconds, 2) for sandbox, seconds in by_sandbox.items()
        },
    )
