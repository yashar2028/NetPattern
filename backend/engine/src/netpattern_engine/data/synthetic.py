"""Learnable toy data for every task type, generated deterministically from (seed, index).

Used for tests, sanity runs and the "custom CNN trains on FakeData" check.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from netpattern_engine.spec.data import SyntheticDataset

Box = tuple[float, float, float, float]


@dataclass
class SyntheticItem:
    label: object = None
    boxes: tuple[Box, ...] = ()
    box_labels: tuple[int, ...] = ()


def names(task_type: str, spec: SyntheticDataset) -> tuple[list[str], list[str]]:
    if task_type == "regression":
        return [], ["brightness"]
    if task_type == "segmentation.semantic":
        return ["background"] + [f"shape_{i}" for i in range(1, spec.num_classes)], []
    return [f"class_{i}" for i in range(spec.num_classes)], []


def _rng(spec: SyntheticDataset, index: int) -> np.random.Generator:
    return np.random.default_rng([spec.seed, index])


def describe(task_type: str, spec: SyntheticDataset, index: int) -> SyntheticItem:
    """Labels/boxes without rendering pixels (used while indexing)."""
    return render(task_type, spec, index)[1]


def render(task_type: str, spec: SyntheticDataset, index: int):
    """(image (C,H,W) float32 in [0,1], item, mask or None)."""
    rng = _rng(spec, index)
    size, k = spec.image_size, spec.num_classes
    image = rng.normal(0.25, 0.05, size=(size, size)).astype(np.float32)
    mask = None
    item = SyntheticItem()

    if task_type == "classification.single_label":
        label = int(rng.integers(k))
        image += _stripes(size, label + 1)
        item.label = label
    elif task_type == "classification.multi_label":
        present = tuple(int(i) for i in range(k) if rng.random() < 0.4)
        for label in present:
            image += 0.5 * _band(size, label, k)
        item.label = present
    elif task_type == "regression":
        level = float(rng.uniform(0.0, 1.0))
        y0, x0 = rng.integers(0, size // 2, size=2)
        image[y0 : y0 + size // 2, x0 : x0 + size // 2] += level
        item.label = (level,)
    elif task_type == "segmentation.semantic":
        mask = np.zeros((size, size), dtype=np.int64)
        for _ in range(int(rng.integers(1, 4))):
            label = int(rng.integers(1, k))
            y0, x0 = rng.integers(0, size * 3 // 4, size=2)
            h, w = rng.integers(size // 8, size // 3, size=2)
            mask[y0 : y0 + h, x0 : x0 + w] = label
        image += mask.astype(np.float32) / max(k - 1, 1) * 0.6
    elif task_type == "detection.bbox":
        boxes, labels = [], []
        for _ in range(int(rng.integers(1, 4))):
            label = int(rng.integers(k))
            h, w = rng.integers(size // 6, size // 3, size=2)
            y0 = int(rng.integers(0, size - h))
            x0 = int(rng.integers(0, size - w))
            image[y0 : y0 + h, x0 : x0 + w] = 0.3 + 0.7 * (label + 1) / k
            boxes.append((float(x0), float(y0), float(x0 + w), float(y0 + h)))
            labels.append(label + 1)
        item.boxes, item.box_labels = tuple(boxes), tuple(labels)
    else:
        raise ValueError(f"unknown task type {task_type}")

    image = np.clip(image, 0.0, 1.0)
    channels = np.repeat(image[None], spec.channels, axis=0)
    return channels.astype(np.float32), item, mask


def _stripes(size: int, frequency: int) -> np.ndarray:
    x = np.linspace(0, 2 * np.pi * frequency, size, dtype=np.float32)
    return (0.35 * (np.sin(x) > 0)).astype(np.float32)[None, :].repeat(size, axis=0)


def _band(size: int, label: int, count: int) -> np.ndarray:
    band = np.zeros((size, size), dtype=np.float32)
    height = max(size // count, 1)
    band[label * height : (label + 1) * height, :] = 1.0
    return band
