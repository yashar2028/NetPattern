"""Turn a saved pipeline into an engine request for the worker.

In API specs a dataset node names a `dataset_version_id` (plus an optional `subpath`
inside it) instead of a folder. Here those are checked for ownership and replaced
by the folder the worker expects, so users can only ever reach their own data.
"""

import copy
from pathlib import PurePosixPath
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.dataset import Dataset, DatasetVersion
from app.models.user import User


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
    if version is None or dataset is None or dataset.owner_id != user.id:
        raise RunRequestError(f"dataset version '{version_id}' not found")
    return version


async def resolve_datasets(db: AsyncSession, spec: dict[str, Any], user: User) -> dict[str, Any]:
    spec = copy.deepcopy(spec)
    spec.pop("ui", None)  # canvas layout (node positions) is for the editor only
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
    return spec
