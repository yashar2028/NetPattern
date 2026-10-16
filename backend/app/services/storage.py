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


def predict_inputs_root() -> Path:
    """Images uploaded for prediction jobs: predict-inputs/<user id>/<request id>/."""
    return root() / "predict-inputs"


def _resolve_inside(base: Path, path: str, what: str) -> Path:
    base = base.resolve()
    candidate = Path(path)
    resolved = (candidate if candidate.is_absolute() else base / candidate).resolve()
    if resolved != base and base not in resolved.parents:
        raise ValueError(f"{what} path must be inside {base}: {path}")
    return resolved


def resolve_dataset_path(path: str) -> Path:
    """An absolute path inside the datasets folder; relative paths are taken from there.

    Raises ValueError for anything that escapes the datasets folder.
    """
    return _resolve_inside(datasets_root(), path, "dataset")


def resolve_predict_inputs_path(path: str) -> Path:
    """Like resolve_dataset_path, for the prediction uploads folder."""
    return _resolve_inside(predict_inputs_root(), path, "prediction input")
