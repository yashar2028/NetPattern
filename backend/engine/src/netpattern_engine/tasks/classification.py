from __future__ import annotations

from typing import Any

import torch
from torch import nn

from netpattern_engine.data.index import Sample
from netpattern_engine.evaluation.metrics import MultiLabelMetrics, SingleLabelMetrics
from netpattern_engine.spec.training import TrainingSpec
from netpattern_engine.tasks.base import TaskPlugin, class_counts, resolve_class_weights


class SingleLabelClassification(TaskPlugin):
    type = "classification.single_label"
    monitor = "f1_macro"
    monitor_mode = "max"

    def make_loss(self, training: TrainingSpec, train_samples: list[Sample]) -> nn.Module:
        counts = class_counts(train_samples, self.num_outputs)
        weights = resolve_class_weights(training, counts, self.num_outputs)
        return nn.CrossEntropyLoss(weight=weights, label_smoothing=training.label_smoothing)

    def metrics(self) -> Any:
        return SingleLabelMetrics(self.classes)


class MultiLabelClassification(TaskPlugin):
    type = "classification.multi_label"
    monitor = "f1_macro"
    monitor_mode = "max"

    def make_loss(self, training: TrainingSpec, train_samples: list[Sample]) -> nn.Module:
        pos_weight = None
        if training.class_weights == "auto":
            positives = class_counts(train_samples, self.num_outputs)
            negatives = len(train_samples) - positives
            pos_weight = (negatives / positives.clamp(min=1)).float()
        elif isinstance(training.class_weights, list):
            pos_weight = resolve_class_weights(training, None, self.num_outputs)
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def metrics(self) -> Any:
        return MultiLabelMetrics(self.classes, self.spec.threshold)


class Regression(TaskPlugin):
    """Targets are standardized with train-split statistics; metrics use the real scale."""

    type = "regression"
    monitor = "mae"
    monitor_mode = "min"

    def __init__(self, spec: Any) -> None:
        super().__init__(spec)
        self.mean = torch.zeros(1)
        self.std = torch.ones(1)

    def setup(self, classes: list[str], targets: list[str], train_samples: list[Sample]) -> None:
        super().setup(classes, targets, train_samples)
        count = len(self.targets)
        if len(train_samples) < 2:  # e.g. validating a model in the editor
            self.mean, self.std = torch.zeros(count), torch.ones(count)
            return
        values = torch.tensor([s.label for s in train_samples], dtype=torch.float64)
        self.mean = values.mean(dim=0).float()
        std = values.std(dim=0)
        self.std = torch.where(std > 1e-8, std, torch.ones_like(std)).float()

    @property
    def num_outputs(self) -> int:
        return len(self.targets)

    def state(self) -> dict[str, Any]:
        return {
            **super().state(),
            "target_mean": self.mean.tolist(),
            "target_std": self.std.tolist(),
        }

    def make_loss(self, training: TrainingSpec, train_samples: list[Sample]) -> nn.Module:
        return nn.MSELoss()

    def _normalize(self, targets: torch.Tensor) -> torch.Tensor:
        return (targets - self.mean.to(targets.device)) / self.std.to(targets.device)

    def denormalize(self, outputs: torch.Tensor) -> torch.Tensor:
        return outputs * self.std.to(outputs.device) + self.mean.to(outputs.device)

    def train_step(
        self, model: nn.Module, images: Any, targets: Any, loss_fn: nn.Module
    ) -> torch.Tensor:
        return super().train_step(model, images, self._normalize(targets), loss_fn)

    def eval_step(self, model, images, targets, loss_fn):
        loss, output, _ = super().eval_step(model, images, self._normalize(targets), loss_fn)
        return loss, self.denormalize(output), targets

    def metrics(self) -> Any:
        from netpattern_engine.evaluation.metrics import RegressionMetrics

        return RegressionMetrics(self.targets)
