"""Turn a saved pipeline into an engine request for the worker.

In API specs a dataset node names a `dataset_version_id` (plus an optional `subpath`
inside it) instead of a folder. Here those are checked for ownership and replaced
by the folder the worker expects, so users can only ever reach their own data.
The version ids are kept on the job, so the data a run used can be traced and
protected from deletion while the run is queued or running.
"""

import copy
import re
from pathlib import Path, PurePosixPath
from typing import Any

from sqlalchemy import Text, cast, func, select
from sqlalchemy.dialects.postgresql import ARRAY, array
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.dataset import READY, Dataset, DatasetVersion
from app.models.job import Job, JobStatus
from app.models.user import User
from app.services import storage

JOB_ID = re.compile(r"^job_[0-9a-f]{32}$")
CHECKPOINT_NAMES = ("best", "last")


class RunRequestError(ValueError):
    pass


def dataset_folder(version: DatasetVersion, subpath: str | None) -> str:
    folder = PurePosixPath(version.path)
    if subpath:
        relative = PurePosixPath(subpath)
        if relative.is_absolute() or ".." in relative.parts:
            raise RunRequestError("subpath must be a folder inside the dataset version")
        folder = folder / relative
    return str(folder)


async def owned_dataset_version(db: AsyncSession, version_id: str, user: User) -> DatasetVersion:
    version = await db.get(DatasetVersion, version_id)
    dataset = await db.get(Dataset, version.dataset_id) if version else None
    if (
        version is None
        or dataset is None
        or dataset.owner_id != user.id
        or version.status != READY
        or version.deleted_at is not None
        or dataset.deleted_at is not None
    ):
        raise RunRequestError(f"dataset version '{version_id}' not found")
    return version


async def resolve_datasets(
    db: AsyncSession, spec: dict[str, Any], user: User
) -> tuple[dict[str, Any], list[str]]:
    """The spec with dataset folders, and the ids of the dataset versions it reads."""
    spec = copy.deepcopy(spec)
    spec.pop("ui", None)  # canvas layout (node positions) is for the editor only
    version_ids: list[str] = []
    for node in spec.get("nodes", []):
        if not isinstance(node, dict) or node.get("type") != "dataset":
            continue
        params = node.setdefault("params", {})
        if "root" in params:
            raise RunRequestError(
                f"dataset node '{node.get('id')}': use dataset_version_id (and subpath), not root"
            )
        if params.get("format") == "synthetic":
            continue
        version_id = params.pop("dataset_version_id", None)
        if not version_id:
            raise RunRequestError(f"dataset node '{node.get('id')}' needs a dataset_version_id")
        version = await owned_dataset_version(db, version_id, user)
        params["root"] = dataset_folder(version, params.pop("subpath", None))
        version_ids.append(version.id)
    return spec, version_ids


async def active_jobs_using(db: AsyncSession, version_ids: list[str]) -> int:
    """Number of queued or running jobs that read any of these dataset versions."""
    if not version_ids:
        return 0
    return (
        await db.scalar(
            select(func.count(Job.id)).where(
                Job.status.in_([JobStatus.QUEUED, JobStatus.RUNNING]),
                Job.dataset_version_ids.has_any(cast(array(version_ids), ARRAY(Text))),
            )
        )
        or 0
    )


def checkpoint_path(run_id: str, name: str) -> Path:
    if not JOB_ID.match(run_id) or name not in CHECKPOINT_NAMES:
        raise ValueError(f"invalid checkpoint reference {run_id}/{name}")
    return storage.run_dir(run_id) / "checkpoints" / f"{name}.pt"


def _dataset_roots(job: Job) -> list[str]:
    """Dataset folders a job's payload reads (predictions only read their inputs)."""
    payload = job.payload or {}
    if job.kind == "predict":
        inputs = payload.get("inputs") or {}
        return [inputs["root"]] if inputs.get("source") == "dataset" else []
    roots = [
        node.get("params", {}).get("root")
        for node in payload.get("pipeline", {}).get("nodes", [])
        if isinstance(node, dict) and node.get("type") == "dataset"
    ]
    if isinstance(payload.get("dataset"), dict):
        roots.append(payload["dataset"].get("root"))
    return [root for root in roots if root]


async def missing_inputs(db: AsyncSession, job: Job) -> list[str]:
    """What a rerun of this job would need but no longer exists."""
    missing: list[str] = []
    ids = list(job.dataset_version_ids or [])
    if ids:
        found = set(
            (
                await db.execute(
                    select(DatasetVersion.id)
                    .join(Dataset, Dataset.id == DatasetVersion.dataset_id)
                    .where(
                        DatasetVersion.id.in_(ids),
                        DatasetVersion.status == READY,
                        DatasetVersion.deleted_at.is_(None),
                        Dataset.deleted_at.is_(None),
                    )
                )
            ).scalars()
        )
        missing += [f"dataset version {i} (deleted or purged)" for i in ids if i not in found]
    # Folders as well: this also covers jobs created before version ids were recorded.
    for root in _dataset_roots(job):
        try:
            exists = storage.resolve_dataset_path(root).is_dir()
        except ValueError:
            exists = False
        if not exists:
            missing.append(f"dataset folder {root}")
    if job.kind == "predict":
        payload = job.payload or {}
        inputs = payload.get("inputs") or {}
        if inputs.get("source") == "upload":
            try:
                exists = storage.resolve_predict_inputs_path(inputs.get("root", "")).is_dir()
            except ValueError:
                exists = False
            if not exists:
                missing.append("the uploaded images")
        source, name = payload.get("source_run_id", ""), payload.get("checkpoint", "")
        try:
            exists = checkpoint_path(source, name).is_file()
        except ValueError:
            exists = False
        if not exists:
            missing.append(f"checkpoint '{name}' of run {source}")
    return list(dict.fromkeys(missing))
