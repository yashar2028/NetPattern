from typing import Literal

from pydantic import Field

from netpattern_engine.spec.base import StrictModel


class OptimizerSpec(StrictModel):
    name: Literal["sgd", "adam", "adamw", "rmsprop"] = "adamw"
    lr: float = Field(default=1e-3, gt=0, le=10)
    weight_decay: float = Field(default=0.0, ge=0)
    momentum: float = Field(default=0.9, ge=0, lt=1)
    # Discriminative learning rate: everything except the task head uses lr * factor.
    backbone_lr_factor: float | None = Field(default=None, gt=0, le=1)


class SchedulerSpec(StrictModel):
    name: Literal["none", "step", "cosine", "onecycle", "plateau"] = "none"
    step_size: int = Field(default=10, ge=1)
    gamma: float = Field(default=0.1, gt=0, le=1)
    warmup_epochs: int = Field(default=0, ge=0)
    # plateau: epochs without improvement before the LR is reduced.
    patience: int = Field(default=3, ge=1)


class EarlyStoppingSpec(StrictModel):
    monitor: str | None = None  # default: the task's main metric
    mode: Literal["auto", "min", "max"] = "auto"
    patience: int = Field(default=5, ge=1)
    min_delta: float = Field(default=0.0, ge=0)


class TrainingSpec(StrictModel):
    epochs: int = Field(default=10, ge=1, le=10_000)
    batch_size: int = Field(default=32, ge=1, le=4096)
    optimizer: OptimizerSpec = OptimizerSpec()
    scheduler: SchedulerSpec = SchedulerSpec()
    label_smoothing: float = Field(default=0.0, ge=0, lt=1)
    # "auto": weights inversely proportional to class frequency in the train split.
    class_weights: Literal["none", "auto"] | list[float] = "none"
    # Mixed precision; only effective on CUDA.
    amp: bool = False
    grad_clip_norm: float | None = Field(default=None, gt=0)
    accumulate_steps: int = Field(default=1, ge=1)
    early_stopping: EarlyStoppingSpec | None = None
    # Metric that decides the "best" checkpoint; default: the task's main metric.
    monitor: str | None = None
    seed: int = 42
    deterministic: bool = False
    device: Literal["auto", "cpu", "cuda"] = "auto"
    num_workers: int = Field(default=2, ge=0, le=32)
    # Cap batches per epoch (quick experiments and tests).
    max_batches_per_epoch: int | None = Field(default=None, ge=1)
    # Minimum seconds between two live batch events.
    log_every_seconds: float = Field(default=2.0, ge=0)
