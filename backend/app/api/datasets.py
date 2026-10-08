"""Datasets: upload a zip archive, get an immutable version, profile it."""

import asyncio
import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Query, Response, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, not_found
from app.core.config import settings
from app.core.errors import ApiError
from app.core.pagination import DEFAULT_LIMIT, Page, paginate
from app.db.session import get_db
from app.models.dataset import Dataset, DatasetVersion
from app.models.user import User
from app.schemas.api import DatasetCreate, DatasetOut, DatasetVersionOut, ProfileRequest, RunOut
from app.services import storage
from app.services.runs import RunRequestError, dataset_folder
from app.services.uploads import UploadError, extract_zip
from app.workers import queue

router = APIRouter(prefix="/v1/datasets", tags=["datasets"])
CHUNK = 1024 * 1024


async def owned_dataset(db: AsyncSession, dataset_id: str, user: User) -> Dataset:
    dataset = await db.get(Dataset, dataset_id)
    if dataset is None or dataset.owner_id != user.id:
        raise not_found("Dataset")
    return dataset


def _version_out(version: DatasetVersion) -> DatasetVersionOut:
    folder = storage.datasets_root() / version.path
    top = (
        sorted(p.name + ("/" if p.is_dir() else "") for p in folder.iterdir())
        if folder.is_dir()
        else []
    )
    return DatasetVersionOut.model_validate(version).model_copy(update={"top_level": top[:50]})


async def _versions(db: AsyncSession, dataset_id: str) -> list[DatasetVersion]:
    query = select(DatasetVersion).where(DatasetVersion.dataset_id == dataset_id)
    return list((await db.execute(query.order_by(DatasetVersion.number))).scalars())


@router.post("", response_model=DatasetOut, status_code=status.HTTP_201_CREATED)
async def create_dataset(
    body: DatasetCreate, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
) -> Dataset:
    dataset = Dataset(owner_id=user.id, name=body.name)
    db.add(dataset)
    await db.commit()
    await db.refresh(dataset)
    return dataset


@router.get("", response_model=Page[DatasetOut])
async def list_datasets(
    limit: int = Query(DEFAULT_LIMIT),
    cursor: str | None = None,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    rows, next_cursor = await paginate(
        db, select(Dataset).where(Dataset.owner_id == user.id), Dataset, limit, cursor
    )
    return {"data": rows, "next_cursor": next_cursor}


@router.get("/{dataset_id}", response_model=DatasetOut)
async def get_dataset(
    dataset_id: str, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
) -> DatasetOut:
    dataset = await owned_dataset(db, dataset_id, user)
    versions = [_version_out(v) for v in await _versions(db, dataset.id)]
    return DatasetOut.model_validate(dataset).model_copy(update={"versions": versions})


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dataset(
    dataset_id: str, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
) -> Response:
    dataset = await owned_dataset(db, dataset_id, user)
    await db.delete(dataset)
    await db.commit()
    shutil.rmtree(storage.datasets_root() / "users" / user.id / dataset.id, ignore_errors=True)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/{dataset_id}/versions", response_model=DatasetVersionOut, status_code=status.HTTP_201_CREATED
)
async def upload_version(
    dataset_id: str,
    file: UploadFile = File(..., description="a .zip archive of the dataset folder"),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> DatasetVersionOut:
    dataset = await owned_dataset(db, dataset_id, user)
    number = (
        await db.scalar(
            select(func.max(DatasetVersion.number)).where(DatasetVersion.dataset_id == dataset.id)
        )
        or 0
    ) + 1
    relative = Path("users") / user.id / dataset.id / f"v{number}"
    target = storage.datasets_root() / relative
    uploads = storage.datasets_root() / ".uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    archive = uploads / f"{uuid4().hex}.zip"

    try:
        size = 0
        with archive.open("wb") as out:
            while chunk := await file.read(CHUNK):
                size += len(chunk)
                if size > settings.MAX_UPLOAD_BYTES:
                    raise ApiError(
                        status.HTTP_413_CONTENT_TOO_LARGE,
                        "upload_too_large",
                        "Upload too large",
                        f"the limit is {settings.MAX_UPLOAD_BYTES} bytes",
                    )
                out.write(chunk)
        if target.exists():
            raise ApiError(
                status.HTTP_409_CONFLICT, "version_exists", "Version already exists; retry"
            )
        files, written = await asyncio.to_thread(
            extract_zip, archive, target, settings.MAX_EXTRACTED_BYTES, settings.MAX_UPLOAD_FILES
        )
    except UploadError as error:
        raise ApiError(
            status.HTTP_400_BAD_REQUEST, "invalid_archive", "Invalid archive", str(error)
        ) from error
    finally:
        archive.unlink(missing_ok=True)

    version = DatasetVersion(
        dataset_id=dataset.id,
        number=number,
        path=str(relative.as_posix()),
        files=files,
        size_bytes=written,
    )
    db.add(version)
    await db.commit()
    await db.refresh(version)
    return _version_out(version)


@router.post(
    "/{dataset_id}/versions/{number}/profile",
    response_model=RunOut,
    status_code=status.HTTP_202_ACCEPTED,
)
async def profile_version(
    dataset_id: str,
    number: int,
    body: ProfileRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Dataset statistics (class counts, sizes, intensities, warnings) as a sandbox job."""
    dataset = await owned_dataset(db, dataset_id, user)
    version = await db.scalar(
        select(DatasetVersion).where(
            DatasetVersion.dataset_id == dataset.id, DatasetVersion.number == number
        )
    )
    if version is None:
        raise not_found("Dataset version")
    try:
        root = dataset_folder(version, body.subpath)
    except RunRequestError as error:
        raise ApiError(status.HTTP_422_UNPROCESSABLE_CONTENT, "invalid_run", str(error)) from error
    payload = {"task": body.task, "dataset": {**body.dataset, "root": root}}
    return await queue.enqueue(db, "profile", payload, owner_id=user.id)
