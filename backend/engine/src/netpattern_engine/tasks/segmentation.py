from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional

from netpattern_engine.data.index import Sample
from netpattern_engine.evaluation.metrics import SegmentationMetrics
from netpattern_engine.spec.training import TrainingSpec
from netpattern_engine.tasks.base import TaskPlugin, resolve_class_weights

AUX_LOSS_WEIGHT = 0.5


def _logits(output: Any, size: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Main (and auxiliary) logits upsampled to the mask size."""
    main, aux = (output["out"], output.get("aux")) if isinstance(output, dict) else (output, None)

    def fit(tensor: torch.Tensor | None) -> torch.Tensor | None:
        if tensor is None or tensor.shape[-2:] == size:
            return tensor
        return functional.interpolate(tensor, size=size, mode="bilinear", align_corners=False)

    return fit(main), fit(aux)


class SemanticSegmentation(TaskPlugin):
    type = "segmentation.semantic"
    monitor = "miou"
    monitor_mode = "max"

    def make_loss(self, training: TrainingSpec, train_samples: list[Sample]) -> nn.Module:
        weights = resolve_class_weights(training, None, self.num_outputs)
        if training.class_weights == "auto":
            weights = None  # pixel frequencies are unknown before reading every mask
        return nn.CrossEntropyLoss(
            weight=weights,
            ignore_index=self.spec.ignore_index,
            label_smoothing=training.label_smoothing,
        )

    def train_step(
        self, model: nn.Module, images: Any, targets: Any, loss_fn: nn.Module
    ) -> torch.Tensor:
        main, aux = _logits(model(images), tuple(targets.shape[-2:]))
        loss = loss_fn(main, targets)
        if aux is not None:
            loss = loss + AUX_LOSS_WEIGHT * loss_fn(aux, targets)
        return loss

    def eval_step(self, model, images, targets, loss_fn):
        main, _ = _logits(model(images), tuple(targets.shape[-2:]))
        return loss_fn(main, targets), main.argmax(dim=1), targets

    def metrics(self) -> Any:
        return SegmentationMetrics(self.classes, ignore_index=self.spec.ignore_index)

    def check_output_shape(self, shape: tuple[int, ...]) -> str | None:
        if len(shape) != 3 or shape[0] != self.num_outputs:
            return (
                f"a segmentation model must output {self.num_outputs} x H x W "
                f"(one channel per class), got {list(shape)}"
            )
        return None
