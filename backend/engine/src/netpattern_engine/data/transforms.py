"""Transform palette and preset pipelines, built on torchvision.transforms.v2.

v2 transforms images, masks and boxes together, so one pipeline serves every task.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from pydantic import Field, ValidationError

from netpattern_engine.errors import Issue, SpecError, issues_from_validation_error
from netpattern_engine.spec.base import StrictModel
from netpattern_engine.spec.transforms import TransformOp, TransformsSpec


@dataclass(frozen=True)
class ModelInput:
    """What a model expects: channels, crop size (H, W), resize size, normalization."""

    channels: int
    size: tuple[int, int]
    resize: int
    mean: tuple[float, ...]
    std: tuple[float, ...]


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- palette


class _Probability(StrictModel):
    p: float = Field(default=0.5, ge=0, le=1)


class _Rotation(StrictModel):
    degrees: float = Field(default=15, ge=0, le=180)


class _Affine(StrictModel):
    degrees: float = Field(default=0, ge=0, le=180)
    translate: tuple[float, float] | None = None
    scale: tuple[float, float] | None = None
    shear: float | None = Field(default=None, ge=0, le=90)


class _ColorJitter(StrictModel):
    brightness: float = Field(default=0.0, ge=0)
    contrast: float = Field(default=0.0, ge=0)
    saturation: float = Field(default=0.0, ge=0)
    hue: float = Field(default=0.0, ge=0, le=0.5)


class _Blur(StrictModel):
    kernel_size: int = Field(default=3, ge=1)
    sigma: tuple[float, float] = (0.1, 2.0)


class _ResizedCrop(StrictModel):
    size: int = Field(ge=8, le=2048)
    scale: tuple[float, float] = (0.08, 1.0)


class _Crop(StrictModel):
    size: int = Field(ge=8, le=2048)
    padding: int | None = Field(default=None, ge=0)


class _Size(StrictModel):
    size: int = Field(ge=8, le=2048)


class _Erasing(StrictModel):
    p: float = Field(default=0.5, ge=0, le=1)
    scale: tuple[float, float] = (0.02, 0.33)


class _Noise(StrictModel):
    mean: float = 0.0
    sigma: float = Field(default=0.05, ge=0, le=1)


def _v2():
    from torchvision.transforms import v2

    return v2


TRANSFORM_OPS: dict[str, tuple[type[StrictModel], Callable[[Any], Any], str]] = {
    "RandomHorizontalFlip": (_Probability, lambda p: _v2().RandomHorizontalFlip(p.p), "geometry"),
    "RandomVerticalFlip": (_Probability, lambda p: _v2().RandomVerticalFlip(p.p), "geometry"),
    "RandomRotation": (_Rotation, lambda p: _v2().RandomRotation(p.degrees), "geometry"),
    "RandomAffine": (
        _Affine,
        lambda p: _v2().RandomAffine(
            p.degrees, translate=p.translate, scale=p.scale, shear=p.shear
        ),
        "geometry",
    ),
    "RandomResizedCrop": (
        _ResizedCrop,
        lambda p: _v2().RandomResizedCrop(p.size, scale=p.scale, antialias=True),
        "geometry",
    ),
    "RandomCrop": (_Crop, lambda p: _v2().RandomCrop(p.size, padding=p.padding), "geometry"),
    "CenterCrop": (_Size, lambda p: _v2().CenterCrop(p.size), "geometry"),
    "Resize": (_Size, lambda p: _v2().Resize((p.size, p.size), antialias=True), "geometry"),
    "ColorJitter": (
        _ColorJitter,
        lambda p: _v2().ColorJitter(p.brightness, p.contrast, p.saturation, p.hue),
        "color",
    ),
    "GaussianBlur": (_Blur, lambda p: _v2().GaussianBlur(p.kernel_size, p.sigma), "color"),
    "GaussianNoise": (_Noise, lambda p: _v2().GaussianNoise(p.mean, p.sigma), "color"),
    "RandomErasing": (
        _Erasing,
        lambda p: _v2().RandomErasing(p.p, scale=p.scale),
        "regularization",
    ),
}


def transform_palette() -> list[dict[str, Any]]:
    return [
        {"op": name, "category": category, "params": model.model_json_schema()}
        for name, (model, _, category) in TRANSFORM_OPS.items()
    ]


def instantiate_ops(ops: list[TransformOp], field: str) -> list[Any]:
    built, issues = [], []
    for i, op in enumerate(ops):
        entry = TRANSFORM_OPS.get(op.op)
        if entry is None:
            issues.append(Issue(f"unknown transform '{op.op}'", field=f"{field}[{i}].op"))
            continue
        params_model, factory, _ = entry
        try:
            built.append(factory(params_model.model_validate(op.params)))
        except ValidationError as error:
            issues.extend(issues_from_validation_error(error, prefix=f"{field}[{i}].params"))
        except (TypeError, ValueError) as error:
            issues.append(Issue(str(error), field=f"{field}[{i}].params"))
    if issues:
        raise SpecError(issues)
    return built


# --------------------------------------------------------------------------- presets


def build_transforms(
    spec: TransformsSpec, task_type: str, model_input: ModelInput, train: bool
) -> Any:
    v2 = _v2()
    steps = instantiate_ops(spec.train if train else spec.eval, "train" if train else "eval")
    height, width = model_input.size
    if spec.image_size is not None:
        height = width = spec.image_size
    resize = spec.image_size and round(spec.image_size / 0.875) or model_input.resize

    if spec.preset == "none":
        # torchvision refuses an empty Compose; no steps means images pass through unchanged.
        return v2.Compose(steps or [v2.Identity()])

    normalize = v2.Normalize(list(model_input.mean), list(model_input.std))
    if task_type == "detection.bbox":
        # Detection models resize and normalize internally.
        return v2.Compose([*steps, v2.SanitizeBoundingBoxes()])
    if task_type == "segmentation.semantic":
        return v2.Compose([*steps, v2.Resize((height, width), antialias=True), normalize])
    if spec.preset == "basic":
        return v2.Compose([*steps, v2.Resize((height, width), antialias=True), normalize])
    if train:
        crop = v2.RandomResizedCrop((height, width), scale=(0.7, 1.0), antialias=True)
        return v2.Compose([*steps, crop, normalize])
    return v2.Compose(
        [*steps, v2.Resize(resize, antialias=True), v2.CenterCrop((height, width)), normalize]
    )
