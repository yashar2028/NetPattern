"""Turn a resolved pipeline into data loaders + a model, and run train / sanity / prefetch."""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from netpattern_engine.data.dataset import ImageDataset, load_image
from netpattern_engine.data.index import DatasetIndex, build_index
from netpattern_engine.data.readers import file_kind, require_module
from netpattern_engine.data.splits import split_samples
from netpattern_engine.data.transforms import build_transforms
from netpattern_engine.errors import EngineError
from netpattern_engine.models.build import BuiltModel, build_model
from netpattern_engine.spec.pipeline import ResolvedPipeline
from netpattern_engine.tasks import TaskPlugin, get_task
from netpattern_engine.training.checkpoint import load_checkpoint
from netpattern_engine.training.seed import resolve_device, seed_worker, set_seed
from netpattern_engine.training.trainer import CancelToken, Trainer, evaluate

Emit = Callable[..., None]
SANITY_STEPS = 60


@dataclass
class Prepared:
    pipeline: ResolvedPipeline
    task: TaskPlugin
    index: DatasetIndex
    splits: dict[str, list[int]]
    built: BuiltModel
    device: torch.device
    generator: torch.Generator


def required_packs(pipeline: ResolvedPipeline) -> list[str]:
    """Capability packs this pipeline needs (PLAN D23); does not import optional packages."""
    packs = set(get_task(pipeline.task).required_packs)
    root = getattr(pipeline.data, "root", None)
    if root and Path(root).is_dir():
        for path in Path(root).rglob("*"):
            if path.is_file() and file_kind(path) in ("dicom", "nifti"):
                packs.add("medical")
                break
    return sorted(packs)


def check_packs(pipeline: ResolvedPipeline) -> None:
    modules = {"medical": ("pydicom", "nibabel"), "detection": ("pycocotools",)}
    for pack in required_packs(pipeline):
        for module in modules.get(pack, ()):
            require_module(module, pack, f"This pipeline ({pack} data or task)")


def prepare(pipeline: ResolvedPipeline, emit: Emit, materialize: bool = True) -> Prepared:
    check_packs(pipeline)
    generator = set_seed(pipeline.training.seed, pipeline.training.deterministic)
    device = resolve_device(pipeline.training.device)
    task = get_task(pipeline.task)

    emit("phase", name="indexing dataset")
    index = build_index(pipeline.data, pipeline.task)
    splits = split_samples(index.samples, pipeline.data.split, task.type)
    if not splits["train"]:
        raise EngineError("the training split is empty")
    train_samples = [index.samples[i] for i in splits["train"]]
    task.setup(index.classes, index.targets, train_samples)
    if task.type == "regression":
        if not index.targets:
            raise EngineError("regression needs at least one target")
    elif len(index.classes) < (1 if task.type == "detection.bbox" else 2):
        raise EngineError(f"{task.type} needs at least two classes, found {index.classes}")
    emit(
        "dataset_indexed",
        samples=len(index.samples),
        classes=index.classes,
        targets=index.targets,
        groups=len({s.group for s in index.samples}),
        splits={name: len(members) for name, members in splits.items()},
        warnings=index.warnings,
    )

    data_channels = load_image(train_samples[0], pipeline.data, task.type).shape[0]
    emit("phase", name="building model")
    built = build_model(pipeline.architecture, task, data_channels, materialize=materialize)
    emit(
        "model_built",
        params=built.total_params,
        trainable_params=built.trainable_params,
        input=dataclasses.asdict(built.input),
        head=built.head,
        device=str(device),
        **{key: value for key, value in built.info.items() if key in ("weights_ref", "note")},
    )
    return Prepared(pipeline, task, index, splits, built, device, generator)


def make_loader(prepared: Prepared, split: str, train: bool) -> Any | None:
    members = prepared.splits.get(split) or []
    if not members:
        return None
    pipeline, task = prepared.pipeline, prepared.task
    transform = build_transforms(pipeline.transforms, task.type, prepared.built.input, train=train)
    dataset = ImageDataset(
        [prepared.index.samples[i] for i in members],
        pipeline.data,
        task.type,
        prepared.built.input.channels,
        task.num_outputs,
        transform,
    )
    spec = pipeline.training
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=spec.batch_size,
        shuffle=train,
        num_workers=spec.num_workers,
        collate_fn=task.collate,
        generator=prepared.generator if train else None,
        worker_init_fn=seed_worker,
        # BatchNorm cannot train on a final batch of one sample.
        drop_last=train and len(dataset) > spec.batch_size,
        pin_memory=prepared.device.type == "cuda",
        persistent_workers=False,
    )


def train(
    pipeline: ResolvedPipeline, run_dir: Path, emit: Emit, cancel: CancelToken
) -> dict[str, Any]:
    prepared = prepare(pipeline, emit)
    task, built = prepared.task, prepared.built
    train_loader = make_loader(prepared, "train", train=True)
    val_loader = make_loader(prepared, "val", train=False)
    if val_loader is None:
        emit(
            "warning",
            message="the validation split is empty; the best checkpoint follows the training loss",
        )
    loss_fn = task.make_loss(
        pipeline.training, [prepared.index.samples[i] for i in prepared.splits["train"]]
    )

    emit("phase", name="training")
    trainer = Trainer(
        task,
        built.module,
        built.head,
        built.input,
        train_loader,
        val_loader,
        pipeline.training,
        prepared.device,
        run_dir,
        emit,
        cancel,
        loss_fn,
    )
    outcome = trainer.fit()

    evaluation: dict[str, Any] = {}
    best_path = run_dir / "checkpoints" / "best.pt"
    if outcome.stopped_reason != "cancelled" and best_path.exists():
        emit("phase", name="evaluating best checkpoint")
        built.module.load_state_dict(load_checkpoint(best_path)["model_state"])
        for split in pipeline.evaluation.splits:
            loader = make_loader(prepared, split, train=False)
            if loader is None:
                continue
            scalars, details = evaluate(
                built.module, loader, task, loss_fn.to(prepared.device), prepared.device
            )
            evaluation[split] = {"metrics": scalars, "details": details}
            emit("evaluation", split=split, metrics=scalars)

    return {
        "status": "cancelled" if outcome.stopped_reason == "cancelled" else "completed",
        "task": task.state(),
        "dataset": {
            "samples": len(prepared.index.samples),
            "splits": {name: len(members) for name, members in prepared.splits.items()},
            "warnings": prepared.index.warnings,
        },
        "model": {
            "params": built.total_params,
            "trainable_params": built.trainable_params,
            "input": dataclasses.asdict(built.input),
            "head": built.head,
            **{key: value for key, value in built.info.items() if key in ("weights_ref", "note")},
        },
        "training": dataclasses.asdict(outcome),
        "evaluation": evaluation,
        "checkpoints": {
            name: f"checkpoints/{name}.pt"
            for name in ("best", "last")
            if (run_dir / "checkpoints" / f"{name}.pt").exists()
        },
    }


def sanity(pipeline: ResolvedPipeline, emit: Emit, steps: int = SANITY_STEPS) -> dict[str, Any]:
    """Overfit one batch: if the loss does not collapse, the pipeline is broken (PLAN §7.6)."""
    prepared = prepare(pipeline, emit)
    task, model = prepared.task, prepared.built.module.to(prepared.device)
    loader = make_loader(prepared, "train", train=False)
    images, targets = next(iter(loader))
    images, targets = task.to_device(images, targets, prepared.device)
    loss_fn = task.make_loss(pipeline.training, []).to(prepared.device)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=pipeline.training.optimizer.lr
    )
    model.train()
    losses = []
    started = time.monotonic()
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = task.train_step(model, images, targets, loss_fn)
        loss.backward()
        optimizer.step()
        losses.append(round(float(loss), 6))
        if step % 10 == 0 or step == steps:
            emit("sanity_step", step=step, steps=steps, loss=losses[-1])
    first, last = losses[0], min(losses[-5:])
    passed = last < task.sanity_ratio * first or last < 0.05
    return {
        "status": "completed",
        "passed": passed,
        "initial_loss": first,
        "final_loss": last,
        "losses": losses,
        "steps": steps,
        "duration_s": round(time.monotonic() - started, 2),
        "advice": (
            None
            if passed
            else "the loss did not drop on a single batch: check labels, "
            "learning rate and frozen layers"
        ),
    }


def prefetch(pipeline: ResolvedPipeline, emit: Emit) -> dict[str, Any]:
    """Download pretrained weights (with network) so training can run offline."""
    task = get_task(pipeline.task)
    task.setup(pipeline.task.classes or ["a", "b"], pipeline.task.targets or ["y"], [])
    channels = 3
    if pipeline.data.format != "synthetic":
        try:
            prepared_index = build_index(pipeline.data, pipeline.task)
            task.setup(prepared_index.classes or ["a", "b"], prepared_index.targets or ["y"], [])
            channels = load_image(prepared_index.samples[0], pipeline.data, task.type).shape[0]
        except EngineError:
            pass
    else:
        channels = pipeline.data.channels
    built = build_model(pipeline.architecture, task, channels, materialize=True)
    weights = built.info.get("weights_ref")
    emit("prefetched", weights=weights)
    return {"status": "completed", "weights": weights}
