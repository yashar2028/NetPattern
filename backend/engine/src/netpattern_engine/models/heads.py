"""Replace a pretrained model's head so it fits the task (PLAN §7.4)."""

from __future__ import annotations

from functools import partial

from torch import nn

from netpattern_engine.errors import EngineError


def set_submodule(model: nn.Module, name: str, module: nn.Module) -> None:
    parent_name, _, child = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, child, module)


def replace_classifier(model: nn.Module, num_outputs: int) -> list[str]:
    """Swap the last Linear (or SqueezeNet's final 1x1 conv) for one with num_outputs."""
    last_linear = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            last_linear = (name, module)
    if last_linear is not None:
        name, module = last_linear
        set_submodule(
            model, name, nn.Linear(module.in_features, num_outputs, bias=module.bias is not None)
        )
        return [name]

    # SqueezeNet classifies with a 1x1 convolution.
    convs = [
        (n, m)
        for n, m in model.named_modules()
        if isinstance(m, nn.Conv2d) and n.startswith("classifier")
    ]
    if convs:
        name, module = convs[-1]
        set_submodule(model, name, nn.Conv2d(module.in_channels, num_outputs, module.kernel_size))
        return [name]
    raise EngineError(f"cannot find the classification head of {type(model).__name__}")


def replace_segmentation_head(
    model: nn.Module, original_classes: int, num_classes: int
) -> list[str]:
    """torchvision FCN/DeepLab/LRASPP: replace the per-class convolutions."""
    replaced = []
    for name, module in list(model.named_modules()):
        if (
            isinstance(module, nn.Conv2d)
            and module.out_channels == original_classes
            and (name.startswith("classifier") or name.startswith("aux_classifier"))
        ):
            set_submodule(
                model,
                name,
                nn.Conv2d(
                    module.in_channels,
                    num_classes,
                    module.kernel_size,
                    module.stride,
                    module.padding,
                ),
            )
            replaced.append(name)
    if not replaced:
        raise EngineError(f"cannot find the segmentation head of {type(model).__name__}")
    return replaced


def replace_detection_head(model: nn.Module, name: str, num_classes: int) -> list[str]:
    """torchvision Faster R-CNN / RetinaNet / FCOS: new class-specific head."""
    if name.startswith("fasterrcnn"):
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return ["roi_heads.box_predictor"]
    if name.startswith("retinanet"):
        from torchvision.models.detection.retinanet import RetinaNetClassificationHead

        norm_layer = partial(nn.GroupNorm, 32) if name.endswith("_v2") else None
        model.head.classification_head = RetinaNetClassificationHead(
            model.backbone.out_channels,
            model.head.classification_head.num_anchors,
            num_classes,
            norm_layer=norm_layer,
        )
        return ["head.classification_head"]
    if name.startswith("fcos"):
        from torchvision.models.detection.fcos import FCOSClassificationHead

        model.head.classification_head = FCOSClassificationHead(
            model.backbone.out_channels, model.head.classification_head.num_anchors, num_classes
        )
        return ["head.classification_head"]
    raise EngineError(f"head replacement is not supported for {name}")


def head_modules(model: nn.Module, names: list[str]) -> list[nn.Module]:
    return [model.get_submodule(name) for name in names]
