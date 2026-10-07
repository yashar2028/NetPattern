from __future__ import annotations

from typing import Any

import torch
from torch import nn

from netpattern_engine.data.index import Sample
from netpattern_engine.evaluation.metrics import DetectionMetrics
from netpattern_engine.spec.training import TrainingSpec
from netpattern_engine.tasks.base import TaskPlugin


class _NoLoss(nn.Module):
    """Detection models compute their losses themselves."""

    def forward(self, *args: Any) -> torch.Tensor:  # pragma: no cover - never called
        raise RuntimeError("detection losses come from the model")


class BoxDetection(TaskPlugin):
    """torchvision detection models: label 0 is background, classes are 1..K."""

    type = "detection.bbox"
    monitor = "map"
    monitor_mode = "max"
    supports_graph = False
    required_packs = ("detection",)
    # The loss sums several terms (RPN, box regression, classification) that fall at
    # different speeds, so it does not collapse as fast as a single cross-entropy.
    sanity_ratio = 0.5

    @property
    def num_outputs(self) -> int:
        return len(self.classes) + 1

    def collate(self, batch: list[tuple[torch.Tensor, Any]]) -> tuple[Any, Any]:
        images, targets = zip(*batch)
        return list(images), list(targets)

    def to_device(self, images: Any, targets: Any, device: torch.device) -> tuple[Any, Any]:
        return (
            [image.to(device) for image in images],
            [{key: value.to(device) for key, value in t.items()} for t in targets],
        )

    def make_loss(self, training: TrainingSpec, train_samples: list[Sample]) -> nn.Module:
        return _NoLoss()

    def train_step(
        self, model: nn.Module, images: Any, targets: Any, loss_fn: nn.Module
    ) -> torch.Tensor:
        losses = model(images, targets)
        return sum(losses.values())

    def eval_step(self, model, images, targets, loss_fn):
        return None, model(images), targets

    def metrics(self) -> Any:
        return DetectionMetrics(self.classes)

    def check_output_shape(self, shape: tuple[int, ...]) -> str | None:
        return None
