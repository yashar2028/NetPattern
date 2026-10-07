"""Optimizers (with discriminative learning rates) and per-batch LR schedules."""

from __future__ import annotations

import math

import torch
from torch import nn

from netpattern_engine.errors import EngineError
from netpattern_engine.spec.training import OptimizerSpec, SchedulerSpec


def build_optimizer(
    model: nn.Module, spec: OptimizerSpec, head: list[str]
) -> torch.optim.Optimizer:
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not trainable:
        raise EngineError("every layer is frozen; unfreeze at least the head")

    def in_head(name: str) -> bool:
        return any(name == h or name.startswith(h + ".") for h in head)

    if spec.backbone_lr_factor is not None and head:
        groups = [
            {"params": [p for n, p in trainable if in_head(n)], "lr": spec.lr},
            {
                "params": [p for n, p in trainable if not in_head(n)],
                "lr": spec.lr * spec.backbone_lr_factor,
            },
        ]
        groups = [g for g in groups if g["params"]]
    else:
        groups = [{"params": [p for _, p in trainable], "lr": spec.lr}]

    if spec.name == "sgd":
        return torch.optim.SGD(
            groups, lr=spec.lr, momentum=spec.momentum, weight_decay=spec.weight_decay
        )
    if spec.name == "adam":
        return torch.optim.Adam(groups, lr=spec.lr, weight_decay=spec.weight_decay)
    if spec.name == "rmsprop":
        return torch.optim.RMSprop(
            groups, lr=spec.lr, momentum=spec.momentum, weight_decay=spec.weight_decay
        )
    return torch.optim.AdamW(groups, lr=spec.lr, weight_decay=spec.weight_decay)


class LRSchedule:
    """Calls after every optimizer step and after every epoch, whatever the policy."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        spec: SchedulerSpec,
        epochs: int,
        steps_per_epoch: int,
        monitor_mode: str,
    ) -> None:
        self.spec = spec
        self.optimizer = optimizer
        self.steps_per_epoch = max(steps_per_epoch, 1)
        total = epochs * self.steps_per_epoch
        warmup = spec.warmup_epochs * self.steps_per_epoch
        self.batch_scheduler = None
        self.epoch_scheduler = None

        if spec.name == "onecycle":
            self.batch_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=[g["lr"] for g in optimizer.param_groups], total_steps=total
            )
        elif spec.name == "plateau":
            self.epoch_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode=monitor_mode, factor=spec.gamma, patience=spec.patience
            )
        else:

            def factor(step: int) -> float:
                if warmup and step < warmup:
                    return (step + 1) / warmup
                if spec.name == "step":
                    return spec.gamma ** ((step // self.steps_per_epoch) // spec.step_size)
                if spec.name == "cosine":
                    progress = (step - warmup) / max(total - warmup, 1)
                    return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
                return 1.0

            self.batch_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)

    def after_step(self) -> None:
        if self.batch_scheduler is not None:
            try:
                self.batch_scheduler.step()
            except ValueError:  # OneCycle past its last step (e.g. resumed runs)
                pass

    def after_epoch(self, monitor_value: float | None) -> None:
        if self.epoch_scheduler is not None and monitor_value is not None:
            self.epoch_scheduler.step(monitor_value)

    @property
    def lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])
