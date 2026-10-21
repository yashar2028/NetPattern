"""Deeper looks at a model (PLAN §7, Phase 5): error gallery and learning-rate finder."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from netpattern_engine.data.dataset import load_image
from netpattern_engine.tasks.base import main_output

Emit = Callable[..., None]

ERROR_LIMIT = 24
ERRORS_DIR = "errors"


def save_png(array: np.ndarray, path: Path) -> None:
    """(C, H, W) floats in [0, 1] → 8-bit PNG (grey for 1 channel, RGB from the first 3)."""
    from PIL import Image

    pixels = (np.clip(array, 0, 1) * 255).round().astype(np.uint8)
    pixels = pixels[0] if pixels.shape[0] == 1 else pixels[:3].transpose(1, 2, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(path)


def error_gallery(
    prepared: Any, loader: Any, split: str, run_dir: Path, limit: int = ERROR_LIMIT
) -> dict[str, Any]:
    """Misclassified samples of one split (single-label classification), most confident first.

    Thumbnails are the pixels the model read (before resizing), so they work for every format
    (DICOM, NIfTI slices, synthetic data) without serving dataset files.
    """
    task, model, device = prepared.task, prepared.built.module, prepared.device
    members = prepared.splits[split]
    model.eval()
    wrong: list[tuple[float, int, int, int]] = []
    position = 0
    with torch.no_grad():
        for images, targets in loader:
            images, targets = task.to_device(images, targets, device)
            scores = torch.softmax(main_output(model(images)).float(), dim=1)
            confidence, predicted = scores.max(dim=1)
            for offset in range(len(targets)):
                if int(predicted[offset]) != int(targets[offset]):
                    wrong.append(
                        (
                            float(confidence[offset]),
                            members[position + offset],
                            int(targets[offset]),
                            int(predicted[offset]),
                        )
                    )
            position += len(targets)
    wrong.sort(reverse=True)
    samples = []
    for number, (score, index, true, predicted) in enumerate(wrong[:limit]):
        sample = prepared.index.samples[index]
        image = f"{ERRORS_DIR}/{number}.png"
        save_png(load_image(sample, prepared.pipeline.data, task.type), run_dir / image)
        samples.append(
            {
                "image": image,
                "source": sample.path if sample.index is None else f"{sample.path}#{sample.index}",
                "true": task.classes[true],
                "predicted": task.classes[predicted],
                "score": round(score, 4),
            }
        )
    return {"split": split, "total_errors": len(wrong), "evaluated": position, "samples": samples}


def lr_find(
    pipeline: Any,
    emit: Emit,
    cancel: Any,
    steps: int = 100,
    start_lr: float = 1e-7,
    end_lr: float = 1.0,
) -> dict[str, Any]:
    """Exponentially raise the learning rate over a few batches and watch the loss (L. Smith).

    The suggestion is the learning rate where the smoothed loss falls fastest.
    """
    from netpattern_engine.pipeline import make_loader, prepare
    from netpattern_engine.training.optim import build_optimizer

    prepared = prepare(pipeline, emit)
    task, built, device = prepared.task, prepared.built, prepared.device
    model = built.module.to(device)
    loader = make_loader(prepared, "train", train=True)
    train_samples = [prepared.index.samples[i] for i in prepared.splits["train"]]
    loss_fn = task.make_loss(pipeline.training, train_samples).to(device)
    spec = pipeline.training.optimizer.model_copy(update={"lr": start_lr})
    optimizer = build_optimizer(model, spec, built.head)
    factor = (end_lr / start_lr) ** (1 / max(steps - 1, 1))

    emit("phase", name="finding the learning rate")
    model.train()
    lrs, losses, smoothed, best = [], [], 0.0, math.inf
    started, batches = time.monotonic(), iter(loader)
    for step in range(steps):
        if cancel.cancelled:
            break
        try:
            images, targets = next(batches)
        except StopIteration:
            batches = iter(loader)
            images, targets = next(batches)
        images, targets = task.to_device(images, targets, device)
        optimizer.zero_grad(set_to_none=True)
        loss = task.train_step(model, images, targets, loss_fn)
        loss.backward()
        optimizer.step()
        value = float(loss.detach())
        smoothed = 0.9 * smoothed + 0.1 * value if step else value
        lr = optimizer.param_groups[0]["lr"]
        lrs.append(lr)
        losses.append(round(smoothed, 6))
        best = min(best, smoothed)
        if step % 10 == 0:
            emit("lr_step", step=step, steps=steps, lr=lr, loss=round(smoothed, 6))
        if not math.isfinite(smoothed) or smoothed > 4 * best:
            break  # the loss exploded: higher rates are useless
        for group in optimizer.param_groups:
            group["lr"] *= factor

    suggestion = None
    if len(losses) > 5:
        logs = np.log10(np.array(lrs))
        slopes = np.gradient(np.array(losses), logs)
        lowest = int(np.argmin(losses))
        suggestion = float(lrs[int(np.argmin(slopes[: max(lowest, 1)]))])
    return {
        "status": "completed",
        "lrs": lrs,
        "losses": losses,
        "suggested_lr": suggestion,
        "steps": len(lrs),
        "duration_s": round(time.monotonic() - started, 2),
    }
