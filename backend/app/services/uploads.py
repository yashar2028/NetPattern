"""Safe extraction of uploaded dataset archives (PLAN §12.1), and cleanup after crashes.

Rejects paths that escape the target folder, symlinks, too many files and archives
that expand beyond the size limit. A single top-level folder is stripped, so a zip
of `pets/…` and a zip of `pets`'s contents give the same layout.
"""

import asyncio
import shutil
import stat
import time
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.dataset import FAILED, READY, UPLOADING, DatasetVersion
from app.services import storage

SKIPPED = ("__MACOSX/", ".DS_Store", "Thumbs.db")
CHUNK = 1024 * 1024
PARTIAL_SUFFIX = ".partial"


class UploadError(ValueError):
    pass


def staging_folder(target: Path) -> Path:
    return target.with_name(target.name + PARTIAL_SUFFIX)


def _skip(name: str) -> bool:
    return name.startswith(SKIPPED[0]) or PurePosixPath(name).name in SKIPPED[1:]


def extract_zip(archive: Path, target: Path, max_bytes: int, max_files: int) -> tuple[int, int]:
    """Extract into `target` (created here). Returns (files, bytes)."""
    try:
        zip_file = zipfile.ZipFile(archive)
    except zipfile.BadZipFile as error:
        raise UploadError("the upload is not a valid zip archive") from error

    with zip_file:
        entries = [e for e in zip_file.infolist() if not e.is_dir() and not _skip(e.filename)]
        if not entries:
            raise UploadError("the archive contains no files")
        if len(entries) > max_files:
            raise UploadError(f"the archive has {len(entries)} files; the limit is {max_files}")
        if sum(e.file_size for e in entries) > max_bytes:
            raise UploadError(f"the archive expands beyond the {max_bytes} byte limit")

        parts = [PurePosixPath(e.filename).parts for e in entries]
        tops = {p[0] for p in parts}
        strip = len(tops) == 1 and all(len(p) > 1 for p in parts)

        staging = staging_folder(target)
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        root = staging.resolve()
        written = 0
        try:
            for entry, entry_parts in zip(entries, parts):
                if stat.S_ISLNK(entry.external_attr >> 16):
                    raise UploadError(f"symbolic links are not allowed: {entry.filename}")
                relative = PurePosixPath(*entry_parts[1:]) if strip else PurePosixPath(*entry_parts)
                destination = (root / relative).resolve()
                if root not in destination.parents:
                    raise UploadError(f"unsafe path in archive: {entry.filename}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                with zip_file.open(entry) as source, destination.open("wb") as out:
                    while chunk := source.read(CHUNK):
                        written += len(chunk)
                        if written > max_bytes:  # sizes in the zip header can lie
                            raise UploadError("the archive expands beyond the size limit")
                        out.write(chunk)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        staging.rename(target)
        return len(entries), written


# --------------------------------------------------------------------------- crash cleanup


def _remove_leftovers(active: set[str], known: set[str], cutoff: float) -> None:
    """Delete old upload archives, staging folders and version folders without a row.

    `active`: folders of uploads in progress; `known`: folders of every live version.
    Anything modified after `cutoff` (a timestamp) is left alone.
    """
    datasets = storage.datasets_root()
    for archive in (datasets / ".uploads").glob("*.zip"):
        if archive.stat().st_mtime < cutoff:
            archive.unlink(missing_ok=True)
    for folder in datasets.glob("users/*/*/v*"):
        if not folder.is_dir() or folder.stat().st_mtime >= cutoff:
            continue
        relative = folder.relative_to(datasets).as_posix()
        if relative.endswith(PARTIAL_SUFFIX):
            if relative.removesuffix(PARTIAL_SUFFIX) in active:
                continue
        elif relative in known:
            continue
        shutil.rmtree(folder, ignore_errors=True)


async def sweep_stale_uploads(db: AsyncSession, stale_seconds: int) -> list[str]:
    """Mark interrupted uploads as failed and remove the files they left behind.

    Returns the ids of the versions marked as failed.
    """
    cutoff = datetime.now(UTC) - timedelta(seconds=stale_seconds)
    stale = (
        (
            await db.execute(
                select(DatasetVersion)
                .where(DatasetVersion.status == UPLOADING, DatasetVersion.created_at < cutoff)
                .with_for_update(skip_locked=True)
            )
        )
        .scalars()
        .all()
    )
    for version in stale:
        version.status = FAILED
    await db.commit()

    rows = (
        await db.execute(
            select(DatasetVersion.path, DatasetVersion.status).where(
                DatasetVersion.status.in_([UPLOADING, READY])
            )
        )
    ).all()
    active = {path for path, status in rows if status == UPLOADING}
    known = {path for path, _ in rows}
    await asyncio.to_thread(_remove_leftovers, active, known, time.time() - stale_seconds)
    return [version.id for version in stale]
