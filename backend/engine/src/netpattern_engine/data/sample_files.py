"""Write small synthetic datasets to disk in real file formats (PNG, DICOM, NIfTI).

Used by tests and by the `make_sample_dataset` job to exercise the medical readers,
slicing and volume-aware splits end to end.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from netpattern_engine.data.readers import require_module


def _blob_volume(
    rng: np.random.Generator, size: int, depth: int, label: int
) -> tuple[np.ndarray, np.ndarray]:
    """A noisy volume in HU-like units with a bright sphere whose size depends on the label."""
    volume = rng.normal(40, 15, size=(size, size, depth)).astype(np.float32)
    mask = np.zeros((size, size, depth), dtype=np.uint8)
    z, y, x = np.meshgrid(np.arange(depth), np.arange(size), np.arange(size), indexing="ij")
    cz, cy, cx = (
        depth // 2,
        rng.integers(size // 3, 2 * size // 3),
        rng.integers(size // 3, 2 * size // 3),
    )
    radius = size * (0.12 + 0.1 * label)
    inside = ((x - cx) ** 2 + (y - cy) ** 2 + ((z - cz) * 2.0) ** 2) < radius**2
    inside = inside.transpose(1, 2, 0)  # (y, x, z) to match volume layout
    volume[inside] += 300 + 200 * label
    mask[inside] = 1
    return volume, mask


def write_nifti_dataset(
    out: str | Path,
    volumes: int = 12,
    size: int = 64,
    depth: int = 16,
    classes: int = 2,
    seed: int = 0,
) -> dict[str, str]:
    """Classification folders (one class per folder) and a segmentation layout, both NIfTI."""
    nib = require_module("nibabel", "medical", "Writing NIfTI files")
    out = Path(out)
    rng = np.random.default_rng(seed)
    affine = np.diag([1.0, 1.0, 2.0, 1.0])
    for i in range(volumes):
        label = i % classes
        volume, mask = _blob_volume(rng, size, depth, label)
        class_dir = out / "classification" / f"class_{label}"
        class_dir.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(volume, affine), class_dir / f"case_{i:03d}.nii.gz")
        for sub in ("images", "masks"):
            (out / "segmentation" / sub).mkdir(parents=True, exist_ok=True)
        nib.save(
            nib.Nifti1Image(volume, affine),
            out / "segmentation" / "images" / f"case_{i:03d}.nii.gz",
        )
        nib.save(
            nib.Nifti1Image(mask, affine), out / "segmentation" / "masks" / f"case_{i:03d}.nii.gz"
        )
    (out / "segmentation" / "classes.txt").write_text("background\nlesion\n", encoding="utf-8")
    return {
        "classification": str(out / "classification"),
        "segmentation": str(out / "segmentation"),
    }


def write_dicom_series(
    out: str | Path,
    patients: int = 6,
    slices: int = 8,
    size: int = 64,
    classes: int = 2,
    seed: int = 0,
) -> str:
    """Single-frame DICOM slices in class folders; slices of one patient share a PatientID."""
    pydicom = require_module("pydicom", "medical", "Writing DICOM files")
    from pydicom.dataset import FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    out = Path(out)
    rng = np.random.default_rng(seed)
    for patient in range(patients):
        label = patient % classes
        volume, _ = _blob_volume(rng, size, slices, label)
        series_uid = generate_uid()
        folder = out / f"class_{label}"
        folder.mkdir(parents=True, exist_ok=True)
        for index in range(slices):
            meta = FileMetaDataset()
            meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"  # CT Image Storage
            meta.MediaStorageSOPInstanceUID = generate_uid()
            meta.TransferSyntaxUID = ExplicitVRLittleEndian
            dataset = pydicom.Dataset()
            dataset.file_meta = meta
            dataset.SOPClassUID = meta.MediaStorageSOPClassUID
            dataset.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
            dataset.PatientID = f"patient-{patient:03d}"
            dataset.SeriesInstanceUID = series_uid
            dataset.Modality = "CT"
            dataset.InstanceNumber = index + 1
            dataset.Rows = dataset.Columns = size
            dataset.SamplesPerPixel = 1
            dataset.PhotometricInterpretation = "MONOCHROME2"
            dataset.BitsAllocated = dataset.BitsStored = 16
            dataset.HighBit = 15
            dataset.PixelRepresentation = 1
            dataset.RescaleSlope = 1
            dataset.RescaleIntercept = -1024
            dataset.WindowCenter = 300
            dataset.WindowWidth = 1000
            stored = np.clip(volume[:, :, index] + 1024, -32768, 32767).astype(np.int16)
            dataset.PixelData = stored.tobytes()
            dataset.save_as(folder / f"p{patient:03d}_s{index:02d}.dcm", enforce_file_format=True)
    return str(out)


def write_png_classification(
    out: str | Path, per_class: int = 10, size: int = 32, classes: int = 2, seed: int = 0
) -> str:
    from PIL import Image

    out = Path(out)
    rng = np.random.default_rng(seed)
    for label in range(classes):
        folder = out / f"class_{label}"
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            image = rng.integers(0, 60, size=(size, size, 3), dtype=np.uint8)
            image[:, label * (size // classes) : (label + 1) * (size // classes)] += 150
            Image.fromarray(image).save(folder / f"img_{i:03d}.png")
    return str(out)
