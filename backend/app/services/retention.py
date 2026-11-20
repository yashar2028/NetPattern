"""Retention: deleted datasets and versions stay restorable for a while, then their files go.

Deleting is a soft delete (`deleted_at`). Once the retention period has passed:
- a deleted dataset is removed with its versions and folder;
- a deleted version keeps its row as "purged" (so its number is never reused, and runs
  that read it can say so), and its folder is removed.
Images uploaded for predictions are removed after their own retention period.
"""

import asyncio
import shutil
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.dataset import PURGED, READY, Dataset, DatasetVersion
from app.models.job import Job, JobStatus
from app.services import storage
from app.services.runs import active_jobs_using

DAY_SECONDS = 24 * 60 * 60


def purge_after(deleted_at: datetime | None) -> datetime | None:
    """When something deleted at `deleted_at` stops being restorable."""
    if deleted_at is None:
        return None
    return deleted_at + timedelta(days=settings.DATASET_RETENTION_DAYS)


def _remove(folders: list[Path]) -> None:
    for folder in folders:
        shutil.rmtree(folder, ignore_errors=True)


def _remove_old_inputs(active: set[str], cutoff: float) -> list[str]:
    root = storage.predict_inputs_root()
    removed = []
    for folder in root.glob("*/*"):
        relative = folder.relative_to(root).as_posix()
        if not folder.is_dir() or relative in active or folder.stat().st_mtime >= cutoff:
            continue
        shutil.rmtree(folder, ignore_errors=True)
        removed.append(relative)
    return removed


async def purge_expired(
    db: AsyncSession, dataset_retention_seconds: int, inputs_retention_seconds: int
) -> list[str]:
    """Remove what has been deleted for longer than the retention periods.

    The database is updated first: a crash before the files are gone leaves folders
    without rows, which the upload sweep removes later. Returns what was purged.
    """
    cutoff = datetime.now(UTC) - timedelta(seconds=dataset_retention_seconds)
    purged: list[str] = []
    folders: list[Path] = []

    datasets = (
        (
            await db.execute(
                select(Dataset)
                .where(Dataset.deleted_at.is_not(None), Dataset.deleted_at < cutoff)
                .with_for_update(skip_locked=True)
            )
        )
        .scalars()
        .all()
    )
    for dataset in datasets:
        version_ids = list(
            (
                await db.execute(
                    select(DatasetVersion.id).where(DatasetVersion.dataset_id == dataset.id)
                )
            ).scalars()
        )
        if await active_jobs_using(db, version_ids):
            continue
        folders.append(storage.datasets_root() / "users" / dataset.owner_id / dataset.id)
        purged.append(dataset.id)
        await db.delete(dataset)

    versions = (
        (
            await db.execute(
                select(DatasetVersion)
                .where(
                    DatasetVersion.deleted_at.is_not(None),
                    DatasetVersion.deleted_at < cutoff,
                    DatasetVersion.status == READY,
                )
                .with_for_update(skip_locked=True)
            )
        )
        .scalars()
        .all()
    )
    for version in versions:
        if version.dataset_id in purged or await active_jobs_using(db, [version.id]):
            continue
        version.status = PURGED
        folders.append(storage.datasets_root() / version.path)
        purged.append(version.id)
    await db.commit()
    await asyncio.to_thread(_remove, folders)

    payloads = (
        await db.execute(
            select(Job.payload).where(
                Job.kind == "predict", Job.status.in_([JobStatus.QUEUED, JobStatus.RUNNING])
            )
        )
    ).scalars()
    active = {
        (payload.get("inputs") or {}).get("root", "")
        for payload in payloads
        if isinstance(payload, dict)
    }
    purged += await asyncio.to_thread(
        _remove_old_inputs, active, time.time() - inputs_retention_seconds
    )
    return purged
