"""Paths inside the shared storage volume (PLAN §3: a Docker volume now, S3 later)."""

from pathlib import Path

from app.core.config import settings


def root() -> Path:
    return Path(settings.STORAGE_ROOT)


def datasets_root() -> Path:
    return root() / "datasets"


def runs_root() -> Path:
    return root() / "runs"


def envs_root() -> Path:
    return root() / "envs"


def snapshots_root() -> Path:
    return root() / "engine-snapshots"


def cache_root() -> Path:
    """Pretrained weights (torch hub, Hugging Face), shared by all sandboxes."""
    return root() / "cache"


def run_dir(job_id: str) -> Path:
    return runs_root() / job_id


def resolve_dataset_path(path: str) -> Path:
    """An absolute path inside the datasets folder; relative paths are taken from there.

    Raises ValueError for anything that escapes the datasets folder.
    """
    base = datasets_root().resolve()
    candidate = Path(path)
    resolved = (candidate if candidate.is_absolute() else base / candidate).resolve()
    if resolved != base and base not in resolved.parents:
        raise ValueError(f"dataset path must be inside {base}: {path}")
    return resolved
