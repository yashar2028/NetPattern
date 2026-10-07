"""Architecture spec -> nn.Module (PLAN §9).

- pretrained: torchvision / timm / netpattern U-Net base, task head, then patches
- graph: a custom layer graph compiled with torch.fx

With `materialize=False` everything is built on the meta device: no weights are
downloaded and no memory is allocated (used for validation in the editor).
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any

from torch import nn

from netpattern_engine.data.transforms import IMAGENET_MEAN, IMAGENET_STD, ModelInput
from netpattern_engine.errors import EngineError, Issue, SpecError
from netpattern_engine.graph.compiler import analyze_graph
from netpattern_engine.meta import meta_init
from netpattern_engine.models import heads
from netpattern_engine.models.patches import apply_patches
from netpattern_engine.spec.architecture import GraphArchitecture, PretrainedArchitecture
from netpattern_engine.tasks.base import TaskPlugin

CLASSIFICATION_TASKS = ("classification.single_label", "classification.multi_label", "regression")
DETECTION_OPTIONS = ("min_size", "max_size", "trainable_backbone_layers", "box_detections_per_img")
SEGMENTATION_DEFAULT_SIZE = 520


@dataclass
class BuiltModel:
    module: nn.Module
    input: ModelInput
    head: list[str] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def total_params(self) -> int:
        return sum(p.numel() for p in self.module.parameters())

    @property
    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.module.parameters() if p.requires_grad)


def symbols_for(task: TaskPlugin) -> dict[str, int]:
    return {"num_outputs": task.num_outputs, "num_classes": task.num_outputs}


def build_model(
    arch: Any, task: TaskPlugin, data_channels: int, materialize: bool = True
) -> BuiltModel:
    if isinstance(arch, GraphArchitecture):
        return _build_graph(arch, task, materialize)
    if not isinstance(arch, PretrainedArchitecture):
        raise EngineError(f"unknown architecture kind {type(arch).__name__}")

    base = arch.base
    with _device(materialize):
        if base.source == "netpattern":
            built = _build_unet(arch, task, data_channels, materialize)
        elif base.source == "timm":
            built = _build_timm(arch, task, data_channels, materialize)
        elif task.type == "segmentation.semantic":
            built = _build_tv_segmentation(arch, task, materialize)
        elif task.type == "detection.bbox":
            built = _build_tv_detection(arch, task, materialize)
        else:
            built = _build_tv_classifier(arch, task, materialize)

        if arch.patches:
            input_shape = (built.input.channels, *built.input.size)
            apply_patches(
                built.module, arch.patches, built.head, input_shape, symbols_for(task), materialize
            )
    built.info.update(
        source=base.source, name=base.name, weights=base.weights if materialize else None
    )
    return built


@contextlib.contextmanager
def _device(materialize: bool):
    if materialize:
        yield
    else:
        with meta_init():
            yield


def _issue(message: str, field_name: str) -> SpecError:
    return SpecError([Issue(message, field=f"architecture.{field_name}")])


# --------------------------------------------------------------------------- graph


def _build_graph(arch: GraphArchitecture, task: TaskPlugin, materialize: bool) -> BuiltModel:
    if not task.supports_graph:
        raise _issue(
            f"custom layer graphs are not available for {task.type}; use a zoo model", "kind"
        )
    result = analyze_graph(arch, symbols_for(task), materialize=materialize)
    issues = list(result.issues)
    if not issues and result.output_shape is not None:
        problem = task.check_output_shape(result.output_shape)
        if problem:
            issues.append(Issue(problem, node_id=result.output))
    if issues:
        raise SpecError(issues)
    channels, height, width = arch.input.shape
    model_input = ModelInput(
        channels, (height, width), height, (0.5,) * channels, (0.5,) * channels
    )
    return BuiltModel(result.module, model_input, head=[], info={"graph": result})


# --------------------------------------------------------------------------- torchvision


def _tv():
    import torchvision

    return torchvision


def _weights_enum(name: str, weights: str | None):
    if weights is None:
        return None
    try:
        enum = _tv().models.get_model_weights(name)
        return enum.verify(weights)
    except (KeyError, ValueError) as error:
        raise _issue(f"unknown weights '{weights}' for {name}", "base.weights") from error


def _check_listed(name: str, module: Any, kind: str) -> None:
    if name not in _tv().models.list_models(module=module):
        raise _issue(f"'{name}' is not a torchvision {kind} model", "base.name")


def _classifier_input(weights: Any) -> ModelInput:
    if weights is not None:
        transform = weights.transforms()
        crop = int(transform.crop_size[0])
        return ModelInput(
            3,
            (crop, crop),
            int(transform.resize_size[0]),
            tuple(transform.mean),
            tuple(transform.std),
        )
    return ModelInput(3, (224, 224), 256, IMAGENET_MEAN, IMAGENET_STD)


def _build_tv_classifier(
    arch: PretrainedArchitecture, task: TaskPlugin, materialize: bool
) -> BuiltModel:
    tv = _tv()
    if task.type not in CLASSIFICATION_TASKS:
        raise _issue(
            f"torchvision classification models cannot be used for {task.type}", "base.name"
        )
    name = arch.base.name
    _check_listed(name, tv.models, "classification")
    weights = _weights_enum(name, arch.base.weights)
    if weights is None:
        # A blank model is built with the right number of outputs from the start.
        model = tv.models.get_model(name, weights=None, num_classes=task.num_outputs)
        head = _head_names(model)
    else:
        model = tv.models.get_model(name, weights=weights if materialize else None)
        if arch.replace_head:
            head = heads.replace_classifier(model, task.num_outputs)
        else:
            head = _head_names(model)
            outputs = len(weights.meta.get("categories") or [])
            if outputs != task.num_outputs:
                raise _issue(
                    f"the pretrained head has {outputs} outputs but the task needs "
                    f"{task.num_outputs}; enable replace_head",
                    "replace_head",
                )
    return BuiltModel(
        model, _classifier_input(weights), head, {"weights_ref": str(weights) if weights else None}
    )


def _head_names(model: nn.Module) -> list[str]:
    """The classifier layer: the last Linear, or SqueezeNet's final classifier conv."""
    linear = [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]
    if linear:
        return linear[-1:]
    convs = [
        n
        for n, m in model.named_modules()
        if isinstance(m, nn.Conv2d) and n.startswith("classifier")
    ]
    return convs[-1:]


def _build_tv_segmentation(
    arch: PretrainedArchitecture, task: TaskPlugin, materialize: bool
) -> BuiltModel:
    tv = _tv()
    name = arch.base.name
    _check_listed(name, tv.models.segmentation, "segmentation")
    options = dict(arch.base.options)
    weights = _weights_enum(name, arch.base.weights)
    if weights is not None:
        original = len(weights.meta["categories"])
        if materialize:
            model = tv.models.get_model(name, weights=weights, weights_backbone=None)
        else:
            # Same layout as the pretrained model (which comes with an auxiliary head).
            aux = {"aux_loss": True} if name.startswith(("fcn", "deeplabv3")) else {}
            model = tv.models.get_model(
                name, weights=None, weights_backbone=None, num_classes=original, **aux
            )
        head = heads.replace_segmentation_head(model, original, task.num_outputs)
        mean, std = tuple(weights.transforms().mean), tuple(weights.transforms().std)
    else:
        backbone = options.get("weights_backbone") if materialize else None
        kwargs = {"aux_loss": options["aux_loss"]} if "aux_loss" in options else {}
        model = tv.models.get_model(
            name, weights=None, weights_backbone=backbone, num_classes=task.num_outputs, **kwargs
        )
        head = [
            n
            for n, m in model.named_modules()
            if isinstance(m, nn.Conv2d) and m.out_channels == task.num_outputs and "classifier" in n
        ]
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    size = int(options.get("image_size", SEGMENTATION_DEFAULT_SIZE))
    return BuiltModel(
        model,
        ModelInput(3, (size, size), size, mean, std),
        head,
        {"weights_ref": str(weights) if weights else None},
    )


def _build_tv_detection(
    arch: PretrainedArchitecture, task: TaskPlugin, materialize: bool
) -> BuiltModel:
    tv = _tv()
    name = arch.base.name
    _check_listed(name, tv.models.detection, "detection")
    if name.startswith(("maskrcnn", "keypointrcnn")):
        raise _issue("instance segmentation and keypoints come in a later phase", "base.name")
    options = {key: arch.base.options[key] for key in DETECTION_OPTIONS if key in arch.base.options}
    weights = _weights_enum(name, arch.base.weights)
    replaceable = name.startswith(("fasterrcnn", "retinanet", "fcos"))
    info: dict[str, Any] = {}
    if weights is not None and replaceable:
        model = tv.models.get_model(
            name, weights=weights if materialize else None, weights_backbone=None, **options
        )
        head = heads.replace_detection_head(model, name, task.num_outputs)
    else:
        backbone = arch.base.options.get(
            "weights_backbone", "DEFAULT" if weights is not None else None
        )
        if weights is not None:
            info["note"] = (
                f"{name} keeps only its pretrained backbone; the detection head starts blank"
            )
        model = tv.models.get_model(
            name,
            weights=None,
            weights_backbone=backbone if materialize else None,
            num_classes=task.num_outputs,
            **options,
        )
        head = []
    size = int(options.get("min_size", 800))
    info["weights_ref"] = str(weights) if weights else None
    return BuiltModel(
        model, ModelInput(3, (size, size), size, IMAGENET_MEAN, IMAGENET_STD), head, info
    )


# --------------------------------------------------------------------------- timm / U-Net


def _fit_stats(values: tuple[float, ...], channels: int) -> tuple[float, ...]:
    if len(values) == channels:
        return tuple(values)
    return (sum(values) / len(values),) * channels


def _build_timm(
    arch: PretrainedArchitecture, task: TaskPlugin, data_channels: int, materialize: bool
) -> BuiltModel:
    import timm

    if task.type not in CLASSIFICATION_TASKS:
        raise _issue(
            f"timm classifiers cannot be used for {task.type}; use the U-Net with a timm encoder",
            "base.source",
        )
    name = arch.base.name
    if not timm.is_model(name.split(".")[0]):
        raise _issue(f"'{name}' is not a timm model", "base.name")
    pretrained = materialize and arch.base.weights is not None
    kwargs: dict[str, Any] = {"in_chans": data_channels}
    if arch.replace_head:
        kwargs["num_classes"] = task.num_outputs
    model = timm.create_model(name, pretrained=pretrained, **kwargs)
    classifier = model.get_classifier()
    head = [n for n, m in model.named_modules() if m is classifier and n]
    config = timm.data.resolve_model_data_config(model)
    _, height, width = config["input_size"]
    resize = int(round(height / config.get("crop_pct", 0.875)))
    model_input = ModelInput(
        data_channels,
        (height, width),
        resize,
        _fit_stats(config["mean"], data_channels),
        _fit_stats(config["std"], data_channels),
    )
    return BuiltModel(
        model, model_input, head, {"weights_ref": f"timm:{name}" if arch.base.weights else None}
    )


def _build_unet(
    arch: PretrainedArchitecture, task: TaskPlugin, data_channels: int, materialize: bool
) -> BuiltModel:
    import timm

    from netpattern_engine.models.unet import UNet

    if arch.base.name != "unet":
        raise _issue("the netpattern source provides 'unet'", "base.name")
    if task.type != "segmentation.semantic":
        raise _issue("U-Net is a segmentation model", "base.name")
    encoder = arch.base.encoder or "resnet34"
    if not timm.is_model(encoder.split(".")[0]):
        raise _issue(f"'{encoder}' is not a timm encoder", "base.encoder")
    pretrained = materialize and arch.base.weights is not None
    model = UNet(encoder, data_channels, task.num_outputs, pretrained)
    config = timm.get_pretrained_cfg(encoder.split(".")[0])
    mean = getattr(config, "mean", IMAGENET_MEAN) if config else IMAGENET_MEAN
    std = getattr(config, "std", IMAGENET_STD) if config else IMAGENET_STD
    size = int(arch.base.options.get("image_size", 256))
    return BuiltModel(
        model,
        ModelInput(
            data_channels,
            (size, size),
            size,
            _fit_stats(mean, data_channels),
            _fit_stats(std, data_channels),
        ),
        ["head"],
        {"weights_ref": f"timm:{encoder}" if arch.base.weights else None},
    )
