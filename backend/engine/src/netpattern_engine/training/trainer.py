"""The training loop (PLAN P4): small, fully ours, events built in."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn

from netpattern_engine.spec.training import TrainingSpec
from netpattern_engine.tasks.base import TaskPlugin
from netpattern_engine.training.checkpoint import save_checkpoint
from netpattern_engine.training.optim import LRSchedule, build_optimizer

Emit = Callable[..., None]


class CancelToken:
    """Set by SIGTERM, or by the worker creating a request file in the run directory.

    The file works through any isolation layer (signals do not cross bubblewrap's
    session boundary), and lets data-loader workers finish their batch.
    """

    def __init__(self, request_file: Path | None = None) -> None:
        self._flag = False
        self.request_file = request_file

    def cancel(self) -> None:
        self._flag = True

    @property
    def cancelled(self) -> bool:
        if not self._flag and self.request_file is not None and self.request_file.exists():
            self._flag = True
        return self._flag


def metric_mode(name: str, default: str = "max") -> str:
    lowered = name.lower()
    if any(word in lowered for word in ("loss", "mae", "mse", "rmse", "hamming", "error")):
        return "min"
    return default


def evaluate(
    model: nn.Module,
    loader: Any,
    task: TaskPlugin,
    loss_fn: nn.Module,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    model.eval()
    metrics = task.metrics()
    total_loss, batches = 0.0, 0
    with torch.no_grad():
        for step, (images, targets) in enumerate(loader):
            if max_batches is not None and step >= max_batches:
                break
            images, targets = task.to_device(images, targets, device)
            loss, preds, targets = task.eval_step(model, images, targets, loss_fn)
            metrics.update(preds, targets)
            if loss is not None:
                total_loss += float(loss)
                batches += 1
    scalars, details = metrics.compute()
    if batches:
        scalars = {"loss": round(total_loss / batches, 6), **scalars}
    return scalars, details


@dataclass
class TrainOutcome:
    epochs_run: int = 0
    stopped_reason: str = "completed"
    best_epoch: int | None = None
    best_value: float | None = None
    monitor: str = ""
    history: list[dict[str, Any]] = field(default_factory=list)
    duration_s: float = 0.0


class Trainer:
    def __init__(
        self,
        task: TaskPlugin,
        model: nn.Module,
        head: list[str],
        model_input: Any,
        train_loader: Any,
        val_loader: Any | None,
        spec: TrainingSpec,
        device: torch.device,
        run_dir: Path,
        emit: Emit,
        cancel: CancelToken,
        loss_fn: nn.Module,
    ) -> None:
        self.task = task
        self.model = model.to(device)
        self.head = head
        self.model_input = model_input
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.spec = spec
        self.device = device
        self.run_dir = run_dir
        self.emit = emit
        self.cancel = cancel
        self.loss_fn = loss_fn.to(device)
        self.checkpoints = run_dir / "checkpoints"

        default_monitor = f"val/{task.monitor}" if val_loader is not None else "train/loss"
        self.monitor = (
            spec.monitor
            or (spec.early_stopping.monitor if spec.early_stopping else None)
            or default_monitor
        )
        if self.monitor.startswith("val/") and val_loader is None:
            self.monitor = "train/loss"
        mode = spec.early_stopping.mode if spec.early_stopping else "auto"
        self.mode = metric_mode(self.monitor, task.monitor_mode) if mode == "auto" else mode

    def _batches_per_epoch(self) -> int:
        batches = len(self.train_loader)
        if self.spec.max_batches_per_epoch is not None:
            batches = min(batches, self.spec.max_batches_per_epoch)
        return batches

    def _improved(self, value: float | None, best: float | None) -> bool:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return False
        if best is None:
            return True
        delta = self.spec.early_stopping.min_delta if self.spec.early_stopping else 0.0
        return value < best - delta if self.mode == "min" else value > best + delta

    def fit(self) -> TrainOutcome:
        spec = self.spec
        optimizer = build_optimizer(self.model, spec.optimizer, self.head)
        batches = self._batches_per_epoch()
        steps_per_epoch = max(math.ceil(batches / spec.accumulate_steps), 1)
        schedule = LRSchedule(optimizer, spec.scheduler, spec.epochs, steps_per_epoch, self.mode)
        use_amp = spec.amp and self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        outcome = TrainOutcome(monitor=self.monitor)
        patience_left = spec.early_stopping.patience if spec.early_stopping else None
        started = time.monotonic()

        for epoch in range(1, spec.epochs + 1):
            epoch_started = time.monotonic()
            self.model.train()
            optimizer.zero_grad(set_to_none=True)
            running_loss, seen_batches, seen_samples = 0.0, 0, 0
            last_event = 0.0

            for step, (images, targets) in enumerate(self.train_loader):
                if step >= batches or self.cancel.cancelled:
                    break
                images, targets = self.task.to_device(images, targets, self.device)
                with torch.autocast(self.device.type, enabled=use_amp):
                    loss = self.task.train_step(self.model, images, targets, self.loss_fn)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"the loss became {float(loss)} at epoch {epoch}, batch {step + 1}; "
                        "try a lower learning rate"
                    )
                scaler.scale(loss / spec.accumulate_steps).backward()
                if (step + 1) % spec.accumulate_steps == 0 or step + 1 == batches:
                    if spec.grad_clip_norm is not None:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), spec.grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    schedule.after_step()

                running_loss += float(loss)
                seen_batches += 1
                seen_samples += len(images)
                now = time.monotonic()
                if now - last_event >= spec.log_every_seconds or step + 1 == batches:
                    last_event = now
                    self.emit(
                        "batch",
                        epoch=epoch,
                        batch=step + 1,
                        batches=batches,
                        loss=round(running_loss / seen_batches, 6),
                        lr=schedule.lr,
                        samples_per_s=round(seen_samples / max(now - epoch_started, 1e-6), 2),
                    )

            if self.cancel.cancelled:
                outcome.stopped_reason = "cancelled"
                self._save("last.pt", epoch, {}, optimizer)
                break

            train_loss = running_loss / max(seen_batches, 1)
            metrics: dict[str, Any] = {"train/loss": round(train_loss, 6)}
            if self.val_loader is not None:
                val_scalars, _ = evaluate(
                    self.model, self.val_loader, self.task, self.loss_fn, self.device
                )
                metrics.update({f"val/{key}": value for key, value in val_scalars.items()})
            value = metrics.get(self.monitor)
            improved = self._improved(value, outcome.best_value)
            if improved:
                outcome.best_value, outcome.best_epoch = value, epoch
                self._save("best.pt", epoch, metrics, None)
                if patience_left is not None:
                    patience_left = spec.early_stopping.patience
            elif patience_left is not None:
                patience_left -= 1
            self._save("last.pt", epoch, metrics, optimizer)
            schedule.after_epoch(value)

            summary = {
                "epoch": epoch,
                "epochs": spec.epochs,
                "metrics": metrics,
                "lr": schedule.lr,
                "duration_s": round(time.monotonic() - epoch_started, 2),
                "best": improved,
            }
            outcome.history.append(summary)
            outcome.epochs_run = epoch
            self.emit("epoch_end", **summary)
            if improved:
                self.emit(
                    "checkpoint_saved", name="best", epoch=epoch, monitor=self.monitor, value=value
                )

            if patience_left is not None and patience_left <= 0:
                outcome.stopped_reason = "early_stopping"
                break

        if outcome.best_epoch is None and outcome.epochs_run:
            # The monitor never produced a value (e.g. NaN): keep the last weights as best.
            self._save("best.pt", outcome.epochs_run, outcome.history[-1]["metrics"], None)
            outcome.best_epoch = outcome.epochs_run
        outcome.duration_s = round(time.monotonic() - started, 2)
        return outcome

    def _save(self, name: str, epoch: int, metrics: dict[str, Any], optimizer: Any) -> None:
        save_checkpoint(
            self.checkpoints / name,
            self.model,
            epoch,
            metrics,
            self.task.state(),
            self.model_input,
            optimizer,
        )
