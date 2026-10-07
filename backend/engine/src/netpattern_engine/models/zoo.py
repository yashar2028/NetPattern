"""Model zoo listing and model cards from library metadata (PLAN §7.4)."""

from __future__ import annotations

import fnmatch
from typing import Any

CLASSIFICATION_TASKS = ["classification.single_label", "classification.multi_label", "regression"]

FAMILIES = (
    "alexnet",
    "convnext",
    "densenet",
    "efficientnet",
    "googlenet",
    "inception",
    "maxvit",
    "mnasnet",
    "mobilenet",
    "regnet",
    "resnext",
    "resnet",
    "shufflenet",
    "squeezenet",
    "swin",
    "vgg",
    "vit",
    "wide_resnet",
    "deeplabv3",
    "fcn",
    "lraspp",
    "fasterrcnn",
    "retinanet",
    "fcos",
    "ssdlite",
    "ssd",
)


def family_of(name: str) -> str:
    base = name.split(".")[0].lower()
    for family in FAMILIES:
        if base.startswith(family):
            return family
    return base.split("_")[0]


def _torchvision_entries() -> list[dict[str, Any]]:
    import torchvision

    groups = [
        (torchvision.models, CLASSIFICATION_TASKS),
        (torchvision.models.segmentation, ["segmentation.semantic"]),
        (torchvision.models.detection, ["detection.bbox"]),
    ]
    entries = []
    for module, tasks in groups:
        for name in torchvision.models.list_models(module=module):
            if name.startswith(("maskrcnn", "keypointrcnn")):
                continue
            entries.append(_torchvision_card(name, tasks))
    return entries


def _torchvision_card(name: str, tasks: list[str]) -> dict[str, Any]:
    import torchvision

    card: dict[str, Any] = {
        "source": "torchvision",
        "name": name,
        "family": family_of(name),
        "tasks": tasks,
        "weights": [],
    }
    try:
        enum = torchvision.models.get_model_weights(name)
    except (KeyError, ValueError):
        return card
    card["weights"] = [member.name for member in enum]
    default = enum.DEFAULT
    meta = default.meta
    metrics = meta.get("_metrics", {})
    first_metrics = next(iter(metrics.values()), {}) if metrics else {}
    card.update(
        default_weights=default.name,
        params=meta.get("num_params"),
        gflops=meta.get("_ops"),
        file_size_mb=meta.get("_file_size"),
        metrics=first_metrics,
        trained_on=next(iter(metrics.keys()), None),
        num_categories=len(meta.get("categories") or []),
        recipe=meta.get("recipe"),
    )
    transform = default.transforms()
    crop = getattr(transform, "crop_size", None) or getattr(transform, "resize_size", None)
    if crop:
        card["input_size"] = int(crop[0])
    return card


def _timm_entries(query: str | None, limit: int) -> list[dict[str, Any]]:
    import timm

    pattern = f"*{query}*" if query else "*"
    names = timm.list_models(pattern, pretrained=True)[: limit * 4]
    entries = []
    for name in names:
        entries.append(_timm_card(name))
        if len(entries) >= limit:
            break
    return entries


def _timm_card(name: str) -> dict[str, Any]:
    import timm

    card: dict[str, Any] = {
        "source": "timm",
        "name": name,
        "family": family_of(name),
        "tasks": CLASSIFICATION_TASKS + ["segmentation.semantic (as U-Net encoder)"],
        "weights": ["DEFAULT"],
        "default_weights": "DEFAULT",
    }
    try:
        config = timm.models.get_pretrained_cfg(name)
    except Exception:
        config = None
    if config is not None:
        card.update(
            input_size=int(config.input_size[-1]) if config.input_size else None,
            license=getattr(config, "license", None),
            num_categories=getattr(config, "num_classes", None),
        )
    return card


def list_models(
    task: str | None = None, source: str | None = None, query: str | None = None, limit: int = 50
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if source in (None, "torchvision"):
        entries.extend(_torchvision_entries())
    if source in (None, "netpattern"):
        entries.append(
            {
                "source": "netpattern",
                "name": "unet",
                "family": "unet",
                "tasks": ["segmentation.semantic"],
                "weights": ["DEFAULT"],
                "default_weights": "DEFAULT",
                "encoder": "any timm model (pretrained or blank)",
            }
        )
    if query:
        entries = [e for e in entries if fnmatch.fnmatch(e["name"], f"*{query}*")]
    if task:
        entries = [e for e in entries if any(t.startswith(task) for t in e["tasks"])]
    if source in (None, "timm") and (task is None or task in CLASSIFICATION_TASKS):
        entries.extend(_timm_entries(query, limit))
    return entries[:limit]


def model_card(source: str, name: str) -> dict[str, Any]:
    if source == "torchvision":
        for entry in _torchvision_entries():
            if entry["name"] == name:
                return entry
    elif source == "timm":
        return _timm_card(name)
    raise KeyError(f"unknown model {source}:{name}")
