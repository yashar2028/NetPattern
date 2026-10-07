from __future__ import annotations

from typing import Any, ClassVar

import torch
from torch import nn

from netpattern_engine.data.index import Sample
from netpattern_engine.spec.task import TaskSpec
from netpattern_engine.spec.training import TrainingSpec


def main_output(output: Any) -> torch.Tensor:
    """Models like GoogLeNet/Inception return extra outputs while training."""
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, dict):
        return output["out"]
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, (tuple, list)):
        return output[0]
    raise TypeError(f"unexpected model output type {type(output).__name__}")


class TaskPlugin:
    """One task type. An instance is set up per run with the dataset's classes."""

    type: ClassVar[str]
    monitor: ClassVar[str]  # main metric, e.g. "f1_macro"
    monitor_mode: ClassVar[str]  # "max" or "min"
    supports_graph: ClassVar[bool] = True
    required_packs: ClassVar[tuple[str, ...]] = ()
    # Sanity run passes when the loss on one batch falls below this fraction of its start.
    sanity_ratio: ClassVar[float] = 0.25

    def __init__(self, spec: TaskSpec) -> None:
        self.spec = spec
        self.classes: list[str] = []
        self.targets: list[str] = []

    # -- setup --------------------------------------------------------------------

    def setup(self, classes: list[str], targets: list[str], train_samples: list[Sample]) -> None:
        self.classes = list(classes)
        self.targets = list(targets)

    @property
    def num_outputs(self) -> int:
        return len(self.classes)

    def state(self) -> dict[str, Any]:
        """Saved with checkpoints so a model can be used without the dataset."""
        return {"type": self.type, "classes": self.classes, "targets": self.targets}

    # -- data ---------------------------------------------------------------------

    def collate(self, batch: list[tuple[torch.Tensor, Any]]) -> tuple[Any, Any]:
        return torch.utils.data.default_collate(batch)

    def to_device(self, images: Any, targets: Any, device: torch.device) -> tuple[Any, Any]:
        return images.to(device, non_blocking=True), targets.to(device, non_blocking=True)

    # -- training -----------------------------------------------------------------

    def make_loss(self, training: TrainingSpec, train_samples: list[Sample]) -> nn.Module:
        raise NotImplementedError

    def train_step(
        self, model: nn.Module, images: Any, targets: Any, loss_fn: nn.Module
    ) -> torch.Tensor:
        return loss_fn(main_output(model(images)), targets)

    def eval_step(
        self, model: nn.Module, images: Any, targets: Any, loss_fn: nn.Module
    ) -> tuple[torch.Tensor | None, Any, Any]:
        output = main_output(model(images))
        return loss_fn(output, targets), output, targets

    def metrics(self) -> Any:
        raise NotImplementedError

    def check_output_shape(self, shape: tuple[int, ...]) -> str | None:
        """Shape of one sample's output (no batch dim); a message if it does not fit."""
        if tuple(shape) != (self.num_outputs,):
            return f"the model must output a vector of {self.num_outputs} values, got {list(shape)}"
        return None


def class_counts(samples: list[Sample], num_classes: int) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.float64)
    for sample in samples:
        if isinstance(sample.label, int):
            counts[sample.label] += 1
        elif isinstance(sample.label, tuple):
            for label in sample.label:
                counts[int(label)] += 1
    return counts


def resolve_class_weights(
    training: TrainingSpec, counts: torch.Tensor | None, num_classes: int
) -> torch.Tensor | None:
    if training.class_weights == "none":
        return None
    if isinstance(training.class_weights, list):
        if len(training.class_weights) != num_classes:
            raise ValueError(
                f"class_weights has {len(training.class_weights)} values for {num_classes} classes"
            )
        return torch.tensor(training.class_weights, dtype=torch.float32)
    if counts is None:
        return None
    safe = counts.clamp(min=1)
    weights = safe.sum() / (num_classes * safe)
    return weights.float()
