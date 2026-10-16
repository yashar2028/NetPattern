"""Predictions with a trained checkpoint (job kind "predict").

The model is rebuilt exactly as for training (same pipeline spec, in the same pinned
environment) and the checkpoint's weights are loaded into it. Images get the
pipeline's evaluation transforms; masks and boxes are mapped back to each image's
original size. Volumes give one prediction per slice.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.nn import functional
from torchvision import tv_tensors

from netpattern_engine.data import readers
from netpattern_engine.data.dataset import adapt_channels, load_image
from netpattern_engine.data.index import Sample, expand_image
from netpattern_engine.data.transforms import build_transforms
from netpattern_engine.errors import EngineError
from netpattern_engine.models.build import BuiltModel, build_model
from netpattern_engine.spec.data import SliceSpec
from netpattern_engine.spec.pipeline import ResolvedPipeline
from netpattern_engine.tasks import TaskPlugin, get_task
from netpattern_engine.tasks.base import main_output
from netpattern_engine.training.checkpoint import load_checkpoint
from netpattern_engine.training.seed import resolve_device
from netpattern_engine.training.trainer import CancelToken

Emit = Callable[..., None]
MAX_SAMPLES = 1000
MAX_BATCH = 64
TOP_K = 5
MASKS_DIR = "predictions"


def _round(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def load_model(
    pipeline: ResolvedPipeline, checkpoint: Path, device: torch.device
) -> tuple[TaskPlugin, BuiltModel, dict[str, Any]]:
    if not checkpoint.is_file():
        raise EngineError(f"checkpoint not found: {checkpoint}")
    state = load_checkpoint(checkpoint)
    task = get_task(pipeline.task)
    if state["task"].get("type") != task.type:
        raise EngineError(
            f"the checkpoint is for {state['task'].get('type')}, the pipeline for {task.type}"
        )
    task.load_state(state["task"])
    built = build_model(
        pipeline.architecture, task, int(state["input"]["channels"]), materialize=True
    )
    built.module.load_state_dict(state["model_state"])
    built.module.to(device).eval()
    return task, built, state


def input_samples(root: Path, slices: SliceSpec) -> tuple[list[Sample], list[str]]:
    if not root.is_dir():
        raise EngineError(f"input folder not found: {root}")
    files = sorted(p for p in root.rglob("*") if p.is_file() and readers.is_image_file(p))
    if not files:
        raise EngineError("the inputs contain no supported image files")
    samples: list[Sample] = []
    warnings: list[str] = []
    for path in files:
        samples.extend(expand_image(path, slices))
        if len(samples) > MAX_SAMPLES:
            warnings.append(f"only the first {MAX_SAMPLES} images (or slices) were predicted")
            samples = samples[:MAX_SAMPLES]
            break
    return samples, warnings


def _prepare(
    sample: Sample, pipeline: ResolvedPipeline, task_type: str, channels: int, transform: Any
) -> tuple[torch.Tensor, tuple[int, int]]:
    """The model input for one sample, and the sample's original (height, width)."""
    array = adapt_channels(load_image(sample, pipeline.data, task_type), channels)
    height, width = array.shape[-2:]
    image = tv_tensors.Image(torch.from_numpy(np.ascontiguousarray(array)))
    if task_type == "detection.bbox":
        # Same structure as in training: the detection transforms expect boxes.
        empty = {
            "boxes": tv_tensors.BoundingBoxes(
                torch.zeros((0, 4)), format="XYXY", canvas_size=(height, width)
            ),
            "labels": torch.zeros(0, dtype=torch.int64),
        }
        image, _ = transform(image, empty)
    else:
        image = transform(image)
    return image.as_subclass(torch.Tensor), (int(height), int(width))


def _forward(
    model: torch.nn.Module, task_type: str, tensors: list[torch.Tensor], device: torch.device
) -> list[Any]:
    if task_type == "detection.bbox":
        return list(model([tensor.to(device) for tensor in tensors]))
    if all(tensor.shape == tensors[0].shape for tensor in tensors):
        return list(main_output(model(torch.stack(tensors).to(device))))
    return [main_output(model(tensor[None].to(device)))[0] for tensor in tensors]


def _write_mask(mask: torch.Tensor, path: Path, num_classes: int) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    dtype = np.uint8 if num_classes <= 256 else np.uint16
    Image.fromarray(mask.numpy().astype(dtype)).save(path)


def _describe(
    task: TaskPlugin,
    output: Any,
    size: tuple[int, int],
    model_size: tuple[int, int],
    index: int,
    run_dir: Path,
    score_threshold: float,
) -> dict[str, Any]:
    if task.type == "classification.single_label":
        probabilities = torch.softmax(output.float(), dim=0)
        scores, indices = probabilities.topk(min(TOP_K, probabilities.numel()))
        top = [
            {"label": task.classes[i], "score": _round(s)}
            for s, i in zip(scores.tolist(), indices.tolist())
        ]
        return {"label": top[0]["label"], "score": top[0]["score"], "top": top}

    if task.type == "classification.multi_label":
        probabilities = torch.sigmoid(output.float()).tolist()
        threshold = task.spec.threshold
        return {
            "labels": [c for c, p in zip(task.classes, probabilities) if p >= threshold],
            "scores": {c: _round(p) for c, p in zip(task.classes, probabilities)},
        }

    if task.type == "regression":
        values = task.denormalize(output.float()).tolist()  # type: ignore[attr-defined]
        return {"values": {name: _round(v) for name, v in zip(task.targets, values)}}

    if task.type == "segmentation.semantic":
        logits = output.float()[None]
        if tuple(logits.shape[-2:]) != size:
            logits = functional.interpolate(
                logits, size=size, mode="bilinear", align_corners=False
            )
        mask = logits.argmax(dim=1)[0].cpu()
        name = f"{index:05d}.png"
        _write_mask(mask, run_dir / MASKS_DIR / name, len(task.classes))
        counts = torch.bincount(mask.flatten(), minlength=len(task.classes)).tolist()
        total = mask.numel()
        return {
            "mask": f"{MASKS_DIR}/{name}",
            "class_fractions": {
                task.classes[i]: _round(n / total)
                for i, n in enumerate(counts)
                if n and i < len(task.classes)
            },
        }

    if task.type == "detection.bbox":
        (height, width), (model_height, model_width) = size, model_size
        scale = torch.tensor(
            [width / model_width, height / model_height] * 2, dtype=torch.float32
        )
        boxes = (output["boxes"].float().cpu() * scale).tolist()
        detections = []
        for box, label, score in zip(boxes, output["labels"].tolist(), output["scores"].tolist()):
            if score < score_threshold:
                continue
            name = task.classes[label - 1] if 0 < label <= len(task.classes) else str(label)
            detections.append(
                {"label": name, "score": _round(score), "box": [_round(v, 2) for v in box]}
            )
        return {"detections": detections}

    raise EngineError(f"predictions are not available for {task.type}")


def predict(
    pipeline: ResolvedPipeline,
    checkpoint: Path,
    inputs: Path,
    options: dict[str, Any],
    run_dir: Path,
    emit: Emit,
    cancel: CancelToken,
) -> dict[str, Any]:
    started = time.monotonic()
    device = resolve_device(str(options.get("device", "auto")))
    emit("phase", name="loading model")
    task, built, state = load_model(pipeline, checkpoint, device)
    samples, warnings = input_samples(inputs, pipeline.data.slices)
    transform = build_transforms(pipeline.transforms, task.type, built.input, train=False)
    score_threshold = float(options.get("score_threshold", 0.5))
    batch_size = int(options.get("batch_size", pipeline.training.batch_size))
    batch_size = max(1, min(batch_size, MAX_BATCH))

    emit("phase", name="predicting")
    predictions: list[dict[str, Any]] = []
    status = "completed"
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            if cancel.cancelled:
                status = "cancelled"
                break
            entries: list[dict[str, Any]] = []
            ready: list[tuple[dict[str, Any], torch.Tensor, tuple[int, int], int]] = []
            for index, sample in enumerate(samples[start : start + batch_size], start=start):
                entry: dict[str, Any] = {"file": Path(sample.path).relative_to(inputs).as_posix()}
                if sample.index is not None:
                    entry["slice"] = sample.index
                try:
                    tensor, size = _prepare(
                        sample, pipeline, task.type, built.input.channels, transform
                    )
                except (EngineError, OSError, ValueError) as error:
                    entry["error"] = str(error)
                else:
                    ready.append((entry, tensor, size, index))
                entries.append(entry)
            if ready:
                outputs = _forward(built.module, task.type, [r[1] for r in ready], device)
                for (entry, tensor, size, index), output in zip(ready, outputs):
                    entry.update(
                        _describe(
                            task,
                            output,
                            size,
                            (int(tensor.shape[-2]), int(tensor.shape[-1])),
                            index,
                            run_dir,
                            score_threshold,
                        )
                    )
            predictions.extend(entries)
            emit("predict_progress", done=len(predictions), total=len(samples))

    failed = sum(1 for entry in predictions if "error" in entry)
    if failed:
        warnings.append(f"{failed} image(s) could not be read; see their 'error'")
    return {
        "status": status,
        "task": task.state(),
        "checkpoint": {"epoch": state.get("epoch"), "metrics": state.get("metrics", {})},
        "samples": len(predictions),
        "warnings": warnings,
        "duration_s": round(time.monotonic() - started, 2),
        "predictions": predictions,
    }
