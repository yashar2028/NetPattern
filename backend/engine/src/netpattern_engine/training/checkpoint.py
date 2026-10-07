"""Checkpoints: model weights plus everything needed to use them without the dataset.

Saved with torch.save and loaded with weights_only=True: only tensors and plain
Python values are allowed, so loading never executes code.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Any

import torch
from torch import nn

from netpattern_engine import __version__

FORMAT = "netpattern-checkpoint/1"


def save_checkpoint(
    path: Path,
    model: nn.Module,
    epoch: int,
    metrics: dict[str, Any],
    task_state: dict[str, Any],
    model_input: Any,
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": FORMAT,
        "engine": __version__,
        "epoch": epoch,
        "metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float)) or v is None},
        "task": task_state,
        "input": dataclasses.asdict(model_input),
        "model_state": model.state_dict(),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != FORMAT:
        raise ValueError(f"{path} is not a NetPattern checkpoint")
    return payload
