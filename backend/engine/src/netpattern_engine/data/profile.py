"""Dataset validation and profiling (PLAN §7.2)."""

from __future__ import annotations

import random
from collections import Counter
from typing import Any

import numpy as np

from netpattern_engine.data.dataset import load_image, load_mask
from netpattern_engine.data.index import build_index
from netpattern_engine.data.splits import split_samples
from netpattern_engine.errors import EngineError
from netpattern_engine.spec.task import TaskSpec

SAMPLE_READS = 200
IMBALANCE_WARNING = 10.0


def profile_dataset(data_spec: Any, task: TaskSpec, emit: Any) -> dict[str, Any]:
    index = build_index(data_spec, task)
    splits = split_samples(index.samples, data_spec.split, task.type)
    warnings = list(index.warnings)
    emit(
        "dataset_indexed",
        samples=len(index.samples),
        classes=index.classes,
        splits={k: len(v) for k, v in splits.items()},
    )

    profile: dict[str, Any] = {
        "samples": len(index.samples),
        "groups": len({s.group for s in index.samples}),
        "classes": index.classes,
        "targets": index.targets,
        "splits": {name: len(members) for name, members in splits.items()},
    }

    counts: Counter[int] = Counter()
    for sample in index.samples:
        if isinstance(sample.label, int):
            counts[sample.label] += 1
        elif isinstance(sample.label, tuple) and task.type == "classification.multi_label":
            counts.update(sample.label)
        for label in sample.box_labels:
            counts[label - 1] += 1
    if counts and index.classes:
        per_class = {index.classes[i]: counts.get(i, 0) for i in range(len(index.classes))}
        profile["class_counts"] = per_class
        present = [c for c in per_class.values() if c > 0]
        if present and max(present) / max(min(present), 1) > IMBALANCE_WARNING:
            warnings.append(
                f"classes are imbalanced (largest/smallest = {max(present) / min(present):.1f}); "
                "consider class_weights='auto'"
            )
        tiny = [name for name, count in per_class.items() if count < 5]
        if tiny:
            warnings.append(f"classes with fewer than 5 samples: {', '.join(tiny[:10])}")
    if 0 < len(splits["val"]) < 10:
        warnings.append(
            f"the validation split has only {len(splits['val'])} samples; metrics will be noisy"
        )

    rng = random.Random(0)
    chosen = rng.sample(range(len(index.samples)), min(SAMPLE_READS, len(index.samples)))
    heights, widths, channels, failures = [], [], Counter(), []
    channel_sums, channel_squares, pixel_count = None, None, 0
    mask_values: Counter[int] = Counter()
    for i in chosen:
        sample = index.samples[i]
        try:
            image = load_image(sample, data_spec, task.type)
        except (EngineError, OSError, ValueError) as error:
            failures.append({"path": sample.path, "error": str(error)[:200]})
            continue
        c, h, w = image.shape
        heights.append(h)
        widths.append(w)
        channels[c] += 1
        flat = image.reshape(c, -1).astype(np.float64)
        if channel_sums is None or len(channel_sums) != c:
            channel_sums, channel_squares, pixel_count = np.zeros(c), np.zeros(c), 0
        channel_sums += flat.sum(axis=1)
        channel_squares += (flat**2).sum(axis=1)
        pixel_count += flat.shape[1]
        if task.type == "segmentation.semantic":
            try:
                values, frequencies = np.unique(load_mask(sample, data_spec), return_counts=True)
                mask_values.update(dict(zip(values.tolist(), frequencies.tolist())))
            except (EngineError, OSError, ValueError) as error:
                failures.append({"path": sample.mask or sample.path, "error": str(error)[:200]})

    if heights:
        profile["image_size"] = {
            "height": {
                "min": min(heights),
                "max": max(heights),
                "median": float(np.median(heights)),
            },
            "width": {"min": min(widths), "max": max(widths), "median": float(np.median(widths))},
        }
        profile["channels"] = dict(channels)
    if pixel_count:
        mean = channel_sums / pixel_count
        std = np.sqrt(np.maximum(channel_squares / pixel_count - mean**2, 0))
        profile["intensity"] = {"mean": mean.round(4).tolist(), "std": std.round(4).tolist()}
    if mask_values:
        total = sum(mask_values.values())
        profile["mask_pixels"] = {
            str(value): round(count / total, 4) for value, count in sorted(mask_values.items())
        }
    if failures:
        warnings.append(f"{len(failures)} of {len(chosen)} sampled files could not be read")
        profile["unreadable"] = failures[:20]

    profile["sampled_files"] = len(chosen)
    profile["warnings"] = warnings
    return profile
