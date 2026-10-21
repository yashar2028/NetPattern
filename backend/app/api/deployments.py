"""Deployments and the inference API (PLAN §8.5): serve a model version, call it over HTTP."""

import asyncio
import json
import shutil
import time
from typing import Any

from fastapi import APIRouter, Depends, Query, Request, Response, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import SCOPE_ALL, Caller, get_current_user, get_inference_caller, not_found
from app.api.models import owned_model, owned_version
from app.core.config import settings
from app.core.errors import ApiError
from app.db.session import get_db
from app.models.job import Environment
from app.models.model import (
    Deployment,
    DeploymentStatus,
    InferenceRecord,
    ModelVersion,
)
from app.models.user import User
from app.schemas.api import DeploymentCreate, DeploymentOut, InferenceRecordOut
from app.services import storage, worker
from app.services.inference import (
    ACCEPTED_FORMATS,
    InputError,
    decode_base64,
    limiter,
    write_inputs,
)
from app.services.worker import worker_client

router = APIRouter(prefix="/v1", tags=["deployments"])

OPTIONS = {
    "classification.single_label": {"top_k": "number of ranked labels (default 3)"},
    "classification.multi_label": {"threshold": "minimum score of a returned label"},
    "regression": {},
    "segmentation.semantic": {},
    "detection.bbox": {"threshold": "minimum score of a returned box (default 0.5)"},
}


def _problem(code: int, slug: str, title: str, detail: str | None = None) -> ApiError:
    return ApiError(code, slug, title, detail)


async def owned_deployment(db: AsyncSession, deployment_id: str, user: User) -> Deployment:
    deployment = await db.get(Deployment, deployment_id)
    if deployment is None or deployment.owner_id != user.id:
        raise not_found("Deployment")
    return deployment


async def _account(db: AsyncSession, owner_id: str) -> str | None:
    owner = await db.get(User, owner_id)
    return owner.id if owner is not None and owner.dedicated_worker else None


async def _worker_url(db: AsyncSession, deployment: Deployment) -> str:
    """The deployment's worker; a new one (cold start there) if it is gone."""
    url = await worker.url_of(db, deployment.worker_id)
    if url is None:
        deployment.worker_id, url = await worker.pick(
            db,
            f"deployment:{deployment.id}",
            deployment.hardware_tier,
            await _account(db, deployment.owner_id),
        )
        await db.commit()
    return url


async def _start(db: AsyncSession, deployment: Deployment) -> Deployment:
    """Start the model process and require the parity check to pass."""
    deployment.status, deployment.error = DeploymentStatus.STARTING, None
    await db.commit()
    try:
        async with worker_client(await _worker_url(db, deployment)) as client:
            ok, answer = await worker.post(client, f"/deployments/{deployment.id}/start")
    except ApiError as error:
        ok, answer = False, {"message": error.detail}
    if not ok:
        deployment.status, deployment.error = DeploymentStatus.FAILED, answer.get("message")
    else:
        deployment.parity = answer
        if answer.get("passed") is False:
            deployment.status = DeploymentStatus.FAILED
            deployment.error = "parity check failed: predictions differ from the training run"
        else:
            deployment.status = DeploymentStatus.ACTIVE
    await db.commit()
    await db.refresh(deployment)
    return deployment


async def _stop(db: AsyncSession, deployment: Deployment) -> None:
    url = await worker.url_of(db, deployment.worker_id) or (
        None if deployment.worker_id else settings.WORKER_INTERNAL_URL
    )
    if url is None:
        return  # its worker is gone, and the model process with it
    try:
        async with worker_client(url) as client:
            await worker.post(client, f"/deployments/{deployment.id}/stop")
    except ApiError:
        pass  # the worker is down: there is no process to stop


@router.post("/deployments", response_model=DeploymentOut, status_code=status.HTTP_201_CREATED)
async def create_deployment(
    body: DeploymentCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Deployment:
    await owned_version(db, body.model_version_id, user)
    deployment = Deployment(
        owner_id=user.id,
        model_version_id=body.model_version_id,
        name=body.name,
        mode=body.mode,
        hardware_tier=body.hardware_tier,
    )
    db.add(deployment)
    await db.commit()
    return await _start(db, deployment)


@router.get("/deployments", response_model=list[DeploymentOut])
async def list_deployments(
    user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
) -> list[Deployment]:
    query = (
        select(Deployment)
        .where(Deployment.owner_id == user.id)
        .order_by(Deployment.created_at.desc())
    )
    return list((await db.execute(query)).scalars())


@router.get("/deployments/{deployment_id}", response_model=DeploymentOut)
async def get_deployment(
    deployment_id: str, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
) -> Deployment:
    return await owned_deployment(db, deployment_id, user)


@router.post("/deployments/{deployment_id}/pause", response_model=DeploymentOut)
async def pause_deployment(
    deployment_id: str, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
) -> Deployment:
    deployment = await owned_deployment(db, deployment_id, user)
    await _stop(db, deployment)
    deployment.status = DeploymentStatus.PAUSED
    await db.commit()
    return deployment


@router.post("/deployments/{deployment_id}/resume", response_model=DeploymentOut)
async def resume_deployment(
    deployment_id: str, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
) -> Deployment:
    return await _start(db, await owned_deployment(db, deployment_id, user))


@router.delete("/deployments/{deployment_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_deployment(
    deployment_id: str, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
) -> Response:
    deployment = await owned_deployment(db, deployment_id, user)
    await _stop(db, deployment)
    await db.delete(deployment)
    await db.commit()
    shutil.rmtree(storage.inference_root() / deployment_id, ignore_errors=True)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/deployments/{deployment_id}/requests", response_model=list[InferenceRecordOut])
async def request_log(
    deployment_id: str,
    limit: int = Query(100, ge=1, le=1000),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[InferenceRecord]:
    await owned_deployment(db, deployment_id, user)
    query = (
        select(InferenceRecord)
        .where(InferenceRecord.deployment_id == deployment_id)
        .order_by(InferenceRecord.created_at.desc())
        .limit(limit)
    )
    return list((await db.execute(query)).scalars())


# --------------------------------------------------------------------------- inference


async def _callable_deployment(db: AsyncSession, deployment_id: str, caller: Caller) -> Deployment:
    deployment = await db.get(Deployment, deployment_id)
    if deployment is None or deployment.owner_id != caller.user.id:
        raise not_found("Deployment")
    key = caller.api_key
    if (
        key is not None
        and key.scope != SCOPE_ALL
        and key.deployment_id not in (None, deployment.id)
    ):
        raise _problem(
            status.HTTP_403_FORBIDDEN, "insufficient_scope", "This key is for another deployment"
        )
    return deployment


@router.get("/deployments/{deployment_id}/schema", tags=["inference"])
async def deployment_schema(
    deployment_id: str,
    caller: Caller = Depends(get_inference_caller),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """What the deployment accepts and returns."""
    deployment = await _callable_deployment(db, deployment_id, caller)
    version = await db.get(ModelVersion, deployment.model_version_id)
    task = version.task
    return {
        "deployment_id": deployment.id,
        "model_version": f"{version.model_id}@{version.number}",
        "task": task.get("type"),
        "classes": task.get("classes", []),
        "targets": task.get("targets", []),
        "input": version.input,
        "accepts": ACCEPTED_FORMATS,
        "options": OPTIONS.get(task.get("type"), {}),
        "limits": {
            "images_per_call": settings.INFERENCE_MAX_FILES,
            "bytes_per_call": settings.INFERENCE_MAX_BYTES,
            "calls_per_minute": settings.INFERENCE_RATE_LIMIT_PER_MINUTE,
        },
        "request": {
            "multipart": "fields `files` (one or more images) and optional `options` (JSON)",
            "json": {"inputs": [{"image_base64": "…"}], "options": {}},
        },
    }


async def _read_inputs(request: Request) -> tuple[list[bytes], dict[str, Any]]:
    content_type = request.headers.get("content-type", "")
    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        uploads = [*form.getlist("files"), *form.getlist("images")]
        images = [await upload.read() for upload in uploads if hasattr(upload, "read")]
        raw_options = form.get("options") or "{}"
        try:
            options = json.loads(raw_options) if isinstance(raw_options, str) else {}
        except json.JSONDecodeError as error:
            raise InputError("options must be JSON") from error
    else:
        try:
            body = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise InputError("send multipart/form-data or a JSON body") from error
        inputs = body.get("inputs") if isinstance(body, dict) else None
        if not isinstance(inputs, list):
            raise InputError('the JSON body needs "inputs": [{"image_base64": "…"}]')
        images = [decode_base64(str(item.get("image_base64", ""))) for item in inputs]
        options = body.get("options") or {}
    if not isinstance(options, dict):
        raise InputError("options must be a JSON object")
    if not images:
        raise InputError("no images in the request")
    if len(images) > settings.INFERENCE_MAX_FILES:
        raise InputError(f"at most {settings.INFERENCE_MAX_FILES} images per call")
    if sum(len(image) for image in images) > settings.INFERENCE_MAX_BYTES:
        raise InputError(f"at most {settings.INFERENCE_MAX_BYTES // 1024**2} MB per call")
    return images, options


@router.post("/deployments/{deployment_id}/predict", tags=["inference"])
async def predict(
    deployment_id: str,
    request: Request,
    caller: Caller = Depends(get_inference_caller),
    db: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    started = time.perf_counter()
    deployment = await _callable_deployment(db, deployment_id, caller)
    request_id = getattr(request.state, "request_id", None) or f"req_{int(time.time() * 1000)}"
    record = InferenceRecord(
        deployment_id=deployment.id,
        owner_id=caller.user.id,
        api_key_id=caller.api_key.id if caller.api_key else None,
        status_code=200,
        latency_ms=0.0,
    )
    folder = storage.inference_root() / deployment.id / request_id
    try:
        rate_key = caller.api_key.id if caller.api_key else caller.user.id
        if not limiter.allow(rate_key, settings.INFERENCE_RATE_LIMIT_PER_MINUTE):
            raise _problem(
                status.HTTP_429_TOO_MANY_REQUESTS, "rate_limited", "Too many calls; try again soon"
            )
        if deployment.status != DeploymentStatus.ACTIVE:
            raise _problem(
                status.HTTP_409_CONFLICT,
                "deployment_not_active",
                f"The deployment is {deployment.status}",
                deployment.error,
            )
        try:
            images, options = await _read_inputs(request)
            paths = await asyncio.to_thread(write_inputs, folder, images)
        except InputError as error:
            raise _problem(
                status.HTTP_422_UNPROCESSABLE_CONTENT, "invalid_input", "Invalid input", str(error)
            ) from error
        record.images = len(paths)
        async with worker_client(await _worker_url(db, deployment)) as client:
            ok, answer = await worker.post(
                client,
                f"/deployments/{deployment.id}/predict",
                {"files": paths, "options": options},
            )
        if not ok:
            raise _problem(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                "prediction_failed",
                "The model could not answer",
                answer.get("message"),
            )
        version = await db.get(ModelVersion, deployment.model_version_id)
        environment = await db.get(Environment, version.environment_id)
        record.compute_ms = float(answer.get("timing_ms", {}).get("inference", 0.0))
        total = round((time.perf_counter() - started) * 1000, 2)
        return {
            "request_id": request_id,
            "deployment_id": deployment.id,
            "model_version": f"{version.model_id}@{version.number}",
            "environment_hash": environment.env_hash if environment else None,
            "task": answer.get("task"),
            "outputs": answer.get("outputs", []),
            "timing_ms": {**answer.get("timing_ms", {}), "total": total},
        }
    except ApiError as error:
        record.status_code, record.error = error.status_code, error.detail or error.title
        raise
    finally:
        record.latency_ms = round((time.perf_counter() - started) * 1000, 2)
        db.add(record)
        await db.commit()
        await asyncio.to_thread(shutil.rmtree, folder, True)


@router.get("/models/{model_id}/deployments", response_model=list[DeploymentOut])
async def model_deployments(
    model_id: str, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
) -> list[Deployment]:
    await owned_model(db, model_id, user)
    query = (
        select(Deployment)
        .join(ModelVersion, ModelVersion.id == Deployment.model_version_id)
        .where(ModelVersion.model_id == model_id)
    )
    return list((await db.execute(query)).scalars())
