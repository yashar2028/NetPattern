"""Dataset formats -> a flat list of samples.

Volumes (NIfTI, multi-frame DICOM) expand into one sample per slice. Every sample
carries a `group` (volume, patient or file) so splits never separate slices of the
same scan.
"""

from __future__ import annotations

import csv
import json
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from netpattern_engine.data import readers, synthetic
from netpattern_engine.errors import EngineError
from netpattern_engine.spec.data import (
    CocoDataset,
    CsvDataset,
    ImageFolderDataset,
    SegmentationFoldersDataset,
    SliceSpec,
    SyntheticDataset,
    VocDataset,
    YoloDataset,
)
from netpattern_engine.spec.task import TaskSpec

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class Sample:
    path: str  # image file, or "synthetic:<n>"
    index: int | None = None  # slice/frame of a volume
    label: Any = None  # class index | tuple of class indices | tuple of floats
    mask: str | None = None  # segmentation mask file
    boxes: tuple[Box, ...] = ()  # detection boxes, xyxy in pixels
    box_labels: tuple[int, ...] = ()  # 1..K (0 is background)
    group: str = ""
    split: str | None = None


@dataclass
class DatasetIndex:
    samples: list[Sample]
    classes: list[str] = field(default_factory=list)
    targets: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def build_index(spec: Any, task: TaskSpec) -> DatasetIndex:
    builders = {
        "image_folder": _image_folder,
        "csv": _csv,
        "segmentation_folders": _segmentation_folders,
        "coco": _coco,
        "voc": _voc,
        "yolo": _yolo,
        "synthetic": _synthetic,
    }
    index = builders[spec.format](spec, task)
    if not index.samples:
        raise EngineError("the dataset contains no usable samples")
    if spec.limit is not None and len(index.samples) > spec.limit:
        index.samples = index.samples[: spec.limit]
        index.warnings.append(f"using only the first {spec.limit} samples (limit)")
    if task.classes is not None and task.type != "regression":
        index.classes = list(task.classes)
    return index


# --------------------------------------------------------------------------- helpers


def _root(spec: Any) -> Path:
    root = Path(spec.root)
    if not root.is_dir():
        raise EngineError(f"dataset folder not found: {root}")
    return root


def _image_files(folder: Path) -> list[Path]:
    return sorted(
        path for path in folder.rglob("*") if path.is_file() and readers.is_image_file(path)
    )


def _volume_indices(depth: int, slices: SliceSpec) -> range:
    start = slices.start or 0
    stop = min(slices.stop or depth, depth)
    return range(start, stop, slices.step)


def expand_image(path: Path, slices: SliceSpec, **fields: Any) -> list[Sample]:
    """One sample per 2D image, or one per slice for volumes."""
    kind = readers.file_kind(path)
    group = fields.pop("group", None)
    if kind == "nifti":
        depth = readers.nifti_depth(str(path), slices.axis)
        return [
            Sample(path=str(path), index=i, group=group or str(path), **fields)
            for i in _volume_indices(depth, slices)
        ]
    if kind == "dicom":
        header = readers.read_dicom_header(str(path))
        if header["frames"] > 1:
            return [
                Sample(path=str(path), index=i, group=group or str(path), **fields)
                for i in _volume_indices(header["frames"], slices)
            ]
        # Single-frame files of one patient/series belong together.
        group = group or header["patient_id"] or header["series_uid"] or str(path)
        return [Sample(path=str(path), group=group, **fields)]
    return [Sample(path=str(path), group=group or str(path), **fields)]


def _check_task(spec: Any, task: TaskSpec, allowed: tuple[str, ...]) -> None:
    if task.type not in allowed:
        raise EngineError(f"dataset format '{spec.format}' does not support task '{task.type}'")


# --------------------------------------------------------------------------- formats


def _image_folder(spec: ImageFolderDataset, task: TaskSpec) -> DatasetIndex:
    _check_task(spec, task, ("classification.single_label",))
    root = _root(spec)
    class_dirs = sorted(d for d in root.iterdir() if d.is_dir() and not d.name.startswith("."))
    if not class_dirs:
        raise EngineError(f"{root} has no class subfolders")
    classes = task.classes or [d.name for d in class_dirs]
    position = {name: i for i, name in enumerate(classes)}
    index = DatasetIndex(samples=[], classes=list(classes))
    for class_dir in class_dirs:
        if class_dir.name not in position:
            index.warnings.append(
                f"folder '{class_dir.name}' is not in the task's classes; skipped"
            )
            continue
        for path in _image_files(class_dir):
            index.samples.extend(expand_image(path, spec.slices, label=position[class_dir.name]))
    return index


def _csv(spec: CsvDataset, task: TaskSpec) -> DatasetIndex:
    _check_task(
        spec, task, ("classification.single_label", "classification.multi_label", "regression")
    )
    root = _root(spec)
    manifest = root / spec.manifest
    if not manifest.is_file():
        raise EngineError(f"manifest not found: {manifest}")
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise EngineError(f"{manifest.name} has no rows")

    columns = set(rows[0])
    needed = [spec.path_column]
    if task.type == "regression":
        targets = spec.target_columns or task.targets
        if not targets:
            raise EngineError("regression needs target_columns in the dataset (or task.targets)")
        needed += targets
    else:
        needed.append(spec.label_column)
    for column in (spec.group_column, spec.split_column, spec.slice_column):
        if column:
            needed.append(column)
    missing = [column for column in needed if column not in columns]
    if missing:
        raise EngineError(f"{manifest.name} is missing column(s): {', '.join(missing)}")

    index = DatasetIndex(samples=[])
    if task.type == "regression":
        index.targets = list(targets)
    elif task.type == "classification.single_label":
        index.classes = task.classes or sorted({row[spec.label_column].strip() for row in rows})
    else:
        index.classes = task.classes or sorted(
            {
                label.strip()
                for row in rows
                for label in row[spec.label_column].split(spec.label_separator)
                if label.strip()
            }
        )
    position = {name: i for i, name in enumerate(index.classes)}

    missing_files = 0
    for row_number, row in enumerate(rows, start=2):
        path = root / row[spec.path_column]
        if not path.is_file():
            missing_files += 1
            continue
        try:
            label = _csv_label(row, spec, task, position, index.targets)
        except ValueError as error:
            raise EngineError(f"{manifest.name} line {row_number}: {error}") from error
        fields: dict[str, Any] = {"label": label}
        if spec.group_column and row[spec.group_column]:
            fields["group"] = row[spec.group_column]
        if spec.split_column:
            fields["split"] = _normalize_split(row[spec.split_column])
        if spec.slice_column and row[spec.slice_column].strip():
            group = fields.pop("group", None) or str(path)
            index.samples.append(
                Sample(path=str(path), index=int(row[spec.slice_column]), group=group, **fields)
            )
        else:
            index.samples.extend(expand_image(path, spec.slices, **fields))
    if missing_files:
        index.warnings.append(f"{missing_files} row(s) point to missing files; skipped")
    return index


def _csv_label(
    row: dict[str, str],
    spec: CsvDataset,
    task: TaskSpec,
    position: dict[str, int],
    targets: list[str],
) -> Any:
    if task.type == "regression":
        try:
            return tuple(float(row[column]) for column in targets)
        except ValueError as error:
            raise ValueError(f"target values must be numbers ({error})") from None
    if task.type == "classification.single_label":
        name = row[spec.label_column].strip()
        if name not in position:
            raise ValueError(f"unknown class '{name}'")
        return position[name]
    names = [name.strip() for name in row[spec.label_column].split(spec.label_separator)]
    unknown = [name for name in names if name and name not in position]
    if unknown:
        raise ValueError(f"unknown label(s) {unknown}")
    return tuple(sorted(position[name] for name in names if name))


def _normalize_split(value: str) -> str | None:
    value = value.strip().lower()
    aliases = {"validation": "val", "valid": "val", "training": "train", "testing": "test"}
    value = aliases.get(value, value)
    return value if value in ("train", "val", "test") else None


def _segmentation_folders(spec: SegmentationFoldersDataset, task: TaskSpec) -> DatasetIndex:
    _check_task(spec, task, ("segmentation.semantic",))
    root = _root(spec)
    images_dir, masks_dir = root / spec.images_dir, root / spec.masks_dir
    for folder in (images_dir, masks_dir):
        if not folder.is_dir():
            raise EngineError(f"folder not found: {folder}")
    masks = {readers.file_stem(path): path for path in _image_files(masks_dir)}
    index = DatasetIndex(samples=[], classes=_read_class_file(root))
    unmatched = 0
    for image in _image_files(images_dir):
        mask = masks.get(readers.file_stem(image))
        if mask is None:
            unmatched += 1
            continue
        samples = expand_image(image, spec.slices, mask=str(mask))
        if spec.slices.skip_empty_masks and readers.file_kind(mask) == "nifti":
            samples = [
                s
                for s in samples
                if readers.has_foreground(
                    readers.remap_mask(
                        readers.read_mask_volume_slice(str(mask), spec.slices.axis, s.index or 0),
                        spec.mask_mapping,
                    ),
                    task.ignore_index,
                )
            ]
        index.samples.extend(samples)
    if unmatched:
        index.warnings.append(f"{unmatched} image(s) have no matching mask; skipped")
    if not index.classes and not task.classes:
        index.classes = _infer_mask_classes(index.samples, spec, task.ignore_index)
    return index


def _infer_mask_classes(
    samples: list[Sample], spec: SegmentationFoldersDataset, ignore_index: int
) -> list[str]:
    highest = 0
    for mask in {s.mask for s in samples if s.mask}:
        if readers.file_kind(mask) == "nifti":
            values = np.rint(readers.load_nifti(mask, spec.slices.axis)).astype(np.int64)
        else:
            values = readers.read_mask_natural(mask)
        values = readers.remap_mask(values, spec.mask_mapping)
        present = values[values != ignore_index]
        if present.size:
            highest = max(highest, int(present.max()))
    return ["background"] + [f"class_{i}" for i in range(1, max(highest, 1) + 1)]


def _read_class_file(root: Path) -> list[str]:
    for name in ("classes.txt", "labels.txt"):
        path = root / name
        if path.is_file():
            return [
                line.strip()
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
    data_yaml = root / "data.yaml"
    if data_yaml.is_file():
        return _yolo_yaml_names(data_yaml.read_text(encoding="utf-8"))
    return []


def _yolo_yaml_names(text: str) -> list[str]:
    try:
        import yaml  # available through timm's dependencies

        names = yaml.safe_load(text).get("names", [])
    except Exception:
        return []
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names)]
    return [str(name) for name in names]


def _coco(spec: CocoDataset, task: TaskSpec) -> DatasetIndex:
    _check_task(spec, task, ("detection.bbox",))
    root = _root(spec)
    annotations_path = root / spec.annotations
    if not annotations_path.is_file():
        raise EngineError(f"COCO annotations not found: {annotations_path}")
    data = json.loads(annotations_path.read_text(encoding="utf-8"))
    categories = sorted(data.get("categories", []), key=lambda c: c["id"])
    if not categories:
        raise EngineError("COCO file has no categories")
    label_of = {category["id"]: i + 1 for i, category in enumerate(categories)}
    boxes: dict[int, list[tuple[Box, int]]] = {}
    for annotation in data.get("annotations", []):
        if annotation.get("iscrowd"):
            continue
        x, y, w, h = annotation["bbox"]
        if w <= 1 or h <= 1:
            continue
        boxes.setdefault(annotation["image_id"], []).append(
            ((x, y, x + w, y + h), label_of[annotation["category_id"]])
        )

    index = DatasetIndex(samples=[], classes=[c["name"] for c in categories])
    missing = 0
    for image in data.get("images", []):
        path = _find_image(root, spec.images_dir, image["file_name"])
        if path is None:
            missing += 1
            continue
        objects = boxes.get(image["id"], [])
        index.samples.append(
            Sample(
                path=str(path),
                boxes=tuple(box for box, _ in objects),
                box_labels=tuple(label for _, label in objects),
                group=str(path),
            )
        )
    if missing:
        index.warnings.append(f"{missing} image(s) listed in the COCO file were not found")
    return index


def _find_image(root: Path, images_dir: str, file_name: str) -> Path | None:
    for candidate in (root / images_dir / file_name, root / file_name):
        if candidate.is_file():
            return candidate
    return None


def _voc(spec: VocDataset, task: TaskSpec) -> DatasetIndex:
    _check_task(spec, task, ("detection.bbox",))
    root = _root(spec)
    annotations_dir = root / spec.annotations_dir
    if not annotations_dir.is_dir():
        raise EngineError(f"folder not found: {annotations_dir}")
    parsed = []
    names: set[str] = set()
    for xml_path in sorted(annotations_dir.glob("*.xml")):
        tree = ElementTree.parse(xml_path).getroot()
        file_name = tree.findtext("filename") or f"{xml_path.stem}.jpg"
        objects = []
        for obj in tree.findall("object"):
            name = (obj.findtext("name") or "").strip()
            box = obj.find("bndbox")
            if not name or box is None:
                continue
            coords = tuple(
                float(box.findtext(key) or 0) for key in ("xmin", "ymin", "xmax", "ymax")
            )
            objects.append((name, coords))
            names.add(name)
        parsed.append((file_name, objects))

    classes = task.classes or sorted(names)
    position = {name: i + 1 for i, name in enumerate(classes)}
    index = DatasetIndex(samples=[], classes=list(classes))
    missing = 0
    for file_name, objects in parsed:
        path = _find_image(root, spec.images_dir, file_name)
        if path is None:
            missing += 1
            continue
        kept = [(position[name], box) for name, box in objects if name in position]
        index.samples.append(
            Sample(
                path=str(path),
                boxes=tuple(box for _, box in kept),
                box_labels=tuple(label for label, _ in kept),
                group=str(path),
            )
        )
    if missing:
        index.warnings.append(f"{missing} annotated image(s) were not found")
    return index


def _yolo(spec: YoloDataset, task: TaskSpec) -> DatasetIndex:
    _check_task(spec, task, ("detection.bbox",))
    root = _root(spec)
    images_dir, labels_dir = root / spec.images_dir, root / spec.labels_dir
    for folder in (images_dir, labels_dir):
        if not folder.is_dir():
            raise EngineError(f"folder not found: {folder}")
    classes = task.classes or _read_class_file(root)
    highest = -1
    index = DatasetIndex(samples=[])
    for image in _image_files(images_dir):
        label_file = labels_dir / f"{readers.file_stem(image)}.txt"
        boxes, labels = [], []
        if label_file.is_file():
            height, width = readers.image_size_natural(str(image))
            for line in label_file.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls, cx, cy, w, h = int(parts[0]), *map(float, parts[1:5])
                highest = max(highest, cls)
                boxes.append(
                    (
                        (cx - w / 2) * width,
                        (cy - h / 2) * height,
                        (cx + w / 2) * width,
                        (cy + h / 2) * height,
                    )
                )
                labels.append(cls + 1)
        index.samples.append(
            Sample(path=str(image), boxes=tuple(boxes), box_labels=tuple(labels), group=str(image))
        )
    index.classes = list(classes) or [f"class_{i}" for i in range(highest + 1)]
    if highest >= len(index.classes):
        raise EngineError(
            f"labels use class index {highest} but only {len(index.classes)} classes are named"
        )
    return index


def _synthetic(spec: SyntheticDataset, task: TaskSpec) -> DatasetIndex:
    index = DatasetIndex(samples=[])
    for i in range(spec.num_samples):
        item = synthetic.describe(task.type, spec, i)
        index.samples.append(
            Sample(
                path=f"synthetic:{i}",
                label=item.label,
                boxes=item.boxes,
                box_labels=item.box_labels,
                group=f"synthetic:{i}",
            )
        )
    index.classes, index.targets = synthetic.names(task.type, spec)
    return index
