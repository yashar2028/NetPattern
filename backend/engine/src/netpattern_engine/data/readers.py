"""Read images and volumes into float32 arrays in [0, 1], channels first.

- Natural images (PNG/JPEG/TIFF/WebP/BMP): scaled by their bit depth.
- DICOM: rescale slope/intercept, then the window from the file (or the spec).
- NIfTI: reoriented to RAS and sliced along one axis; intensity scaled per volume.

pydicom and nibabel come from the "medical" capability pack and are imported lazily.
"""

from __future__ import annotations

import functools
import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from netpattern_engine.errors import EngineError, MissingPackError
from netpattern_engine.spec.data import IntensitySpec

NATURAL_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
DICOM_SUFFIXES = {".dcm", ".dicom"}

VOLUME_CACHE_SIZE = 4
EPSILON = 1e-6


def file_kind(path: str | Path) -> str:
    name = str(path).lower()
    if name.endswith(".nii") or name.endswith(".nii.gz"):
        return "nifti"
    suffix = Path(name).suffix
    if suffix in DICOM_SUFFIXES:
        return "dicom"
    if suffix in NATURAL_SUFFIXES:
        return "natural"
    return "unknown"


def file_stem(path: str | Path) -> str:
    """File name without image extensions (handles .nii.gz)."""
    name = Path(path).name
    for suffix in (".nii.gz", ".nii"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def is_image_file(path: str | Path) -> bool:
    return file_kind(path) != "unknown" and not Path(path).name.startswith(".")


def require_module(module: str, pack: str, reason: str) -> Any:
    try:
        return importlib.import_module(module)
    except ImportError as error:
        raise MissingPackError(pack, reason) from error


# --------------------------------------------------------------------------- intensity


def window_bounds(center: float, width: float) -> tuple[float, float]:
    return center - width / 2, center + width / 2


def intensity_bounds(
    raw: np.ndarray, spec: IntensitySpec, file_window: tuple[float, float] | None = None
) -> tuple[float, float]:
    """The (low, high) raw values that map to 0 and 1."""
    mode = spec.mode
    if mode == "auto":
        mode = "window" if file_window is not None else "percentile"
    if mode == "window":
        if spec.window_center is not None and spec.window_width is not None:
            return window_bounds(spec.window_center, spec.window_width)
        if file_window is None:
            raise EngineError("intensity mode 'window' needs window_center and window_width")
        return window_bounds(*file_window)
    if mode == "minmax":
        return float(raw.min()), float(raw.max())
    low, high = np.percentile(raw, spec.percentiles)
    return float(low), float(high)


def apply_bounds(raw: np.ndarray, low: float, high: float) -> np.ndarray:
    scaled = (raw.astype(np.float32) - low) / max(high - low, EPSILON)
    return np.clip(scaled, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- natural


def read_natural(path: str, intensity: IntensitySpec) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        image.load()
        mode = image.mode
        if mode in ("I;16", "I;16B", "I;16L", "I;16N", "I", "F"):
            raw = np.asarray(image).astype(np.float32)
            if intensity.mode != "auto":
                return apply_bounds(raw, *intensity_bounds(raw, intensity))[None]
            if mode == "F" or raw.max() > 65535:
                return apply_bounds(raw, float(raw.min()), float(raw.max()))[None]
            return (raw / 65535.0)[None]
        if mode in ("1", "L", "LA"):
            array = np.asarray(image.convert("L"), dtype=np.float32)[None]
        else:
            array = np.asarray(image.convert("RGB"), dtype=np.float32).transpose(2, 0, 1)

    if intensity.mode != "auto":
        return apply_bounds(array, *intensity_bounds(array, intensity))
    return array / 255.0


def read_mask_natural(path: str) -> np.ndarray:
    """A label image: each pixel value is a class index."""
    from PIL import Image

    with Image.open(path) as image:
        if image.mode not in ("1", "L", "P", "I", "I;16", "I;16B", "I;16L"):
            raise EngineError(
                f"mask {Path(path).name} has mode {image.mode}; masks must be single-channel "
                "label images (one class index per pixel)"
            )
        return np.asarray(image).astype(np.int64)


def image_size_natural(path: str) -> tuple[int, int]:
    """(height, width) without decoding the pixels."""
    from PIL import Image

    with Image.open(path) as image:
        width, height = image.size
    return height, width


# --------------------------------------------------------------------------- DICOM


@dataclass
class DicomPixels:
    raw: np.ndarray  # (frames, H, W) float32 after rescale, or (3, H, W) for color
    window: tuple[float, float] | None
    monochrome1: bool
    color: bool


def _pydicom():
    return require_module("pydicom", "medical", "Reading DICOM files")


def _first(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return (
            float(value[0])
            if hasattr(value, "__len__") and not isinstance(value, str)
            else float(value)
        )
    except (TypeError, ValueError, IndexError):
        return None


def read_dicom_header(path: str) -> dict[str, Any]:
    pydicom = _pydicom()
    dataset = pydicom.dcmread(path, stop_before_pixels=True)
    return {
        "frames": int(getattr(dataset, "NumberOfFrames", 1) or 1),
        "patient_id": str(getattr(dataset, "PatientID", "") or ""),
        "series_uid": str(getattr(dataset, "SeriesInstanceUID", "") or ""),
        "rows": int(getattr(dataset, "Rows", 0) or 0),
        "columns": int(getattr(dataset, "Columns", 0) or 0),
    }


@functools.lru_cache(maxsize=VOLUME_CACHE_SIZE)
def read_dicom(path: str) -> DicomPixels:
    pydicom = _pydicom()
    dataset = pydicom.dcmread(path)
    try:
        pixels = dataset.pixel_array
    except Exception as error:  # compressed transfer syntaxes need extra decoders
        raise EngineError(f"cannot decode pixel data of {Path(path).name}: {error}") from error

    samples_per_pixel = int(getattr(dataset, "SamplesPerPixel", 1) or 1)
    if samples_per_pixel == 3:
        if pixels.ndim != 3:
            raise EngineError(f"{Path(path).name}: multi-frame color DICOM is not supported")
        return DicomPixels(pixels.astype(np.float32).transpose(2, 0, 1), None, False, True)

    slope = _first(getattr(dataset, "RescaleSlope", None)) or 1.0
    intercept = _first(getattr(dataset, "RescaleIntercept", None)) or 0.0
    raw = pixels.astype(np.float32) * slope + intercept
    if raw.ndim == 2:
        raw = raw[None]

    center = _first(getattr(dataset, "WindowCenter", None))
    width = _first(getattr(dataset, "WindowWidth", None))
    window = (center, width) if center is not None and width else None
    monochrome1 = str(getattr(dataset, "PhotometricInterpretation", "")) == "MONOCHROME1"
    return DicomPixels(raw, window, monochrome1, False)


def read_dicom_frame(path: str, frame: int | None, intensity: IntensitySpec) -> np.ndarray:
    pixels = read_dicom(path)
    if pixels.color:
        return pixels.raw / 255.0
    bounds = _dicom_bounds(path, _intensity_key(intensity))
    frames = pixels.raw[frame if frame is not None else 0][None]
    mapped = apply_bounds(frames, *bounds)
    return 1.0 - mapped if pixels.monochrome1 else mapped


@functools.lru_cache(maxsize=64)
def _dicom_bounds(path: str, intensity_key: str) -> tuple[float, float]:
    pixels = read_dicom(path)
    return intensity_bounds(
        pixels.raw, IntensitySpec.model_validate_json(intensity_key), pixels.window
    )


# --------------------------------------------------------------------------- NIfTI


def _nibabel():
    return require_module("nibabel", "medical", "Reading NIfTI files")


def nifti_depth(path: str, axis: int) -> int:
    """Number of slices along the RAS axis, from the header only."""
    nib = _nibabel()
    image = nib.load(path)
    shape = image.shape[:3] if len(image.shape) >= 3 else (*image.shape, 1)
    orientation = nib.orientations.io_orientation(image.affine)
    for source_axis, (target_axis, _) in enumerate(orientation[: len(shape)]):
        if int(target_axis) == axis:
            return int(shape[source_axis])
    return int(shape[axis])


@functools.lru_cache(maxsize=VOLUME_CACHE_SIZE)
def load_nifti(path: str, axis: int) -> np.ndarray:
    """The volume as (slices, H, W) float32 raw values, RAS-oriented."""
    nib = _nibabel()
    image = nib.as_closest_canonical(nib.load(path))
    data = np.asanyarray(image.dataobj, dtype=np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    if data.ndim == 2:
        data = data[..., None]
    data = np.moveaxis(data, axis, 0)
    # Rotate so anatomical "up" points up in the 2D slice.
    return np.ascontiguousarray(np.rot90(data, k=1, axes=(1, 2)))


@functools.lru_cache(maxsize=64)
def _nifti_bounds(path: str, axis: int, intensity_key: str) -> tuple[float, float]:
    return intensity_bounds(
        load_nifti(path, axis), IntensitySpec.model_validate_json(intensity_key)
    )


def read_volume_slices(
    path: str, axis: int, index: int, context: int, intensity: IntensitySpec
) -> np.ndarray:
    """One slice (or 2*context+1 neighbouring slices for 2.5D) as channels."""
    kind = file_kind(path)
    if kind == "nifti":
        volume = load_nifti(path, axis)
        bounds = _nifti_bounds(path, axis, _intensity_key(intensity))
    elif kind == "dicom":
        pixels = read_dicom(path)
        volume = pixels.raw
        bounds = _dicom_bounds(path, _intensity_key(intensity))
    else:
        raise EngineError(f"{Path(path).name} is not a volume")

    depth = volume.shape[0]
    indices = [min(max(i, 0), depth - 1) for i in range(index - context, index + context + 1)]
    mapped = apply_bounds(volume[indices], *bounds)
    if kind == "dicom" and read_dicom(path).monochrome1:
        mapped = 1.0 - mapped
    return mapped


def read_mask_volume_slice(path: str, axis: int, index: int) -> np.ndarray:
    volume = load_nifti(path, axis)
    return np.rint(volume[index]).astype(np.int64)


def remap_mask(mask: np.ndarray, mapping: dict[int, int] | None) -> np.ndarray:
    if not mapping:
        return mask
    out = mask.copy()
    for source, target in mapping.items():
        out[mask == source] = target
    return out


def has_foreground(mask: np.ndarray, ignore_index: int) -> bool:
    return bool(np.any((mask > 0) & (mask != ignore_index)))


def _intensity_key(intensity: IntensitySpec) -> str:
    return json.dumps(intensity.model_dump(mode="json"), sort_keys=True)
