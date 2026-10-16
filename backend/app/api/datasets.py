"""Datasets: upload a zip archive, get an immutable version, profile it.

Deleting a dataset or a version is a soft delete: it can be restored for
DATASET_RETENTION_DAYS, after which the worker removes its files (services.retention).
"""

import asyncio
import shutil
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Query, Response, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, not_found
from app.core.config import settings
from app.core.errors import ApiError
from app.core.pagination import DEFAULT_LIMIT, Page, paginate
from app.db.session import get_db
from app.models.dataset import FAILED, READY, UPLOADING, Dataset, DatasetVersion
from app.models.user import User
from app.schemas.api import DatasetCreate, DatasetOut, DatasetVersionOut, ProfileRequest, RunOut
from app.services import storage
from app.services.retention import purge_after
from app.services.runs import RunRequestError, active_jobs_using, dataset_folder
from app.services.uploads import UploadError, extract_zip, staging_folder
from app.workers import queue

router = APIRouter(prefix="/v1/datasets", tags=["datasets"])
CHUNK = 1024 * 1024
RESERVE_ATTEMPTS = 3


async def owned_dataset(
    db: AsyncSession, dataset_id: str, user: User, deleted: bool | None = False
) -> Dataset:
    """`deleted`: False for live datasets only, True for deleted ones only, None for both."""
    dataset = await db.get(Dataset, dataset_id)
    if (
        dataset is None
        or dataset.owner_id != user.id
        or (deleted is not None and (dataset.deleted_at is not None) != deleted)
    ):
        raise not_found("Dataset")
    return dataset


def _version_out(version: DatasetVersion) -> DatasetVersionOut:
    folder = storage.datasets_root() / version.path
    top = (
        sorted(p.name + ("/" if p.is_dir() else "") for p in folder.iterdir())
        if folder.is_dir()
        else []
    )
    return DatasetVersionOut.model_validate(version).model_copy(
        update={"top_level": top[:50], "purge_after": purge_after(version.deleted_at)}
    )


def _dataset_out(dataset: Dataset, versions: list[DatasetVersion]) -> DatasetOut:
    return DatasetOut.model_validate(dataset).model_copy(
        update={
            "versions": [_version_out(v) for v in versions],
            "purge_after": purge_after(dataset.deleted_at),
        }
    )


async def _versions(
    db: AsyncSession, dataset_id: str, include_deleted: bool = False
) -> list[DatasetVersion]:
    """Usable versions: uploads in progress or failed, and purged versions, are not listed."""
    query = select(DatasetVersion).where(
        DatasetVersion.dataset_id == dataset_id, DatasetVersion.status == READY
    )
    if not include_deleted:
        query = query.where(DatasetVersion.deleted_at.is_(None))
    return list((await db.execute(query.order_by(DatasetVersion.number))).scalars())


async def _version(
    db: AsyncSession, dataset: Dataset, number: int, deleted: bool = False
) -> DatasetVersion:
    version = await db.scalar(
        select(DatasetVersion).where(
            DatasetVersion.dataset_id == dataset.id,
            DatasetVersion.number == number,
            DatasetVersion.status == READY,
            (
                DatasetVersion.deleted_at.is_not(None)
                if deleted
                else DatasetVersion.deleted_at.is_(None)
            ),
        )
    )
    if version is None:
        raise not_found("Dataset version")
    return version


async def _refuse_if_in_use(db: AsyncSession, version_ids: list[str], what: str) -> None:
    if await active_jobs_using(db, version_ids):
        raise ApiError(
            status.HTTP_409_CONFLICT,
            "dataset_in_use",
            "Dataset in use",
            f"queued or running runs read this {what}; cancel them or wait until they finish",
        )


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
    deleted: bool = Query(False, description="list deleted (still restorable) datasets instead"),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    query = select(Dataset).where(
        Dataset.owner_id == user.id,
        Dataset.deleted_at.is_not(None) if deleted else Dataset.deleted_at.is_(None),
    )
    rows, next_cursor = await paginate(db, query, Dataset, limit, cursor)
    return {"data": [_dataset_out(row, []) for row in rows], "next_cursor": next_cursor}


@router.get("/{dataset_id}", response_model=DatasetOut)
async def get_dataset(
    dataset_id: str,
    include_deleted: bool = Query(False, description="also list deleted versions"),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> DatasetOut:
    dataset = await owned_dataset(db, dataset_id, user, deleted=None)
    return _dataset_out(dataset, await _versions(db, dataset.id, include_deleted))


@router.delete("/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_dataset(
    dataset_id: str, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
) -> Response:
    """Soft delete: restorable for DATASET_RETENTION_DAYS, then its files are removed."""
    dataset = await owned_dataset(db, dataset_id, user)
    version_ids = list(
        (
            await db.execute(
                select(DatasetVersion.id).where(DatasetVersion.dataset_id == dataset.id)
            )
        ).scalars()
    )
    await _refuse_if_in_use(db, version_ids, "dataset")
    dataset.deleted_at = datetime.now(UTC)
    await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{dataset_id}/restore", response_model=DatasetOut)
async def restore_dataset(
    dataset_id: str, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
) -> DatasetOut:
    dataset = await owned_dataset(db, dataset_id, user, deleted=True)
    dataset.deleted_at = None
    await db.commit()
    return _dataset_out(dataset, await _versions(db, dataset.id))


@router.delete("/{dataset_id}/versions/{number}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_version(
    dataset_id: str,
    number: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Response:
    """Soft delete one version; restorable for DATASET_RETENTION_DAYS."""
    version = await _version(db, await owned_dataset(db, dataset_id, user), number)
    await _refuse_if_in_use(db, [version.id], "dataset version")
    version.deleted_at = datetime.now(UTC)
    await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/{dataset_id}/versions/{number}/restore", response_model=DatasetVersionOut)
async def restore_version(
    dataset_id: str,
    number: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> DatasetVersionOut:
    version = await _version(db, await owned_dataset(db, dataset_id, user), number, deleted=True)
    version.deleted_at = None
    await db.commit()
    return _version_out(version)


async def _reserve_version(db: AsyncSession, dataset: Dataset, user: User) -> DatasetVersion:
    """Claim the next version number with an "uploading" row before any file is written.

    Failed and interrupted uploads keep their numbers, so a number is never reused.
    """
    dataset_id, owner_id = dataset.id, user.id  # a rollback expires the ORM objects
    for attempt in range(1, RESERVE_ATTEMPTS + 1):
        number = (
            await db.scalar(
                select(func.max(DatasetVersion.number)).where(
                    DatasetVersion.dataset_id == dataset_id
                )
            )
            or 0
        ) + 1
        version = DatasetVersion(
            dataset_id=dataset_id,
            number=number,
            path=(Path("users") / owner_id / dataset_id / f"v{number}").as_posix(),
            files=0,
            size_bytes=0,
            status=UPLOADING,
        )
        db.add(version)
        try:
            await db.commit()
        except IntegrityError:  # a concurrent upload took this number
            await db.rollback()
            if attempt == RESERVE_ATTEMPTS:
                raise ApiError(
                    status.HTTP_409_CONFLICT, "version_conflict", "Concurrent uploads; retry"
                ) from None
            continue
        return version
    raise RuntimeError("unreachable")


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

        version = await _reserve_version(db, dataset, user)
        version_id = version.id
        target = storage.datasets_root() / version.path
        try:
            # Our row owns this number: a folder already there was left by a crashed upload.
            shutil.rmtree(target, ignore_errors=True)
            files, written = await asyncio.to_thread(
                extract_zip,
                archive,
                target,
                settings.MAX_EXTRACTED_BYTES,
                settings.MAX_UPLOAD_FILES,
            )
        except Exception:
            await db.rollback()
            failed = await db.get(DatasetVersion, version_id)
            if failed is not None:
                failed.status = FAILED
                await db.commit()
            shutil.rmtree(staging_folder(target), ignore_errors=True)
            raise
    except UploadError as error:
        raise ApiError(
            status.HTTP_400_BAD_REQUEST, "invalid_archive", "Invalid archive", str(error)
        ) from error
    finally:
        archive.unlink(missing_ok=True)

    version = await db.get(DatasetVersion, version_id, populate_existing=True)
    if version is None:  # the dataset was deleted while the archive was extracted
        shutil.rmtree(target, ignore_errors=True)
        raise not_found("Dataset")
    version.files, version.size_bytes, version.status = files, written, READY
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
    version = await _version(db, dataset, number)
    try:
        root = dataset_folder(version, body.subpath)
    except RunRequestError as error:
        raise ApiError(status.HTTP_422_UNPROCESSABLE_CONTENT, "invalid_run", str(error)) from error
    payload = {"task": body.task, "dataset": {**body.dataset, "root": root}}
    return await queue.enqueue(
        db, "profile", payload, owner_id=user.id, dataset_version_ids=[version.id]
    )
