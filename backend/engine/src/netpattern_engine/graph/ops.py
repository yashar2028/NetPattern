"""The op palette (PLAN §7.5). Users never type in_channels / in_features: each op
receives the shape of its input and builds itself to fit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

from pydantic import Field
from torch import nn

from netpattern_engine.graph import blocks
from netpattern_engine.spec.base import StrictModel

Shape = tuple[int, ...]
IntOrPair = int | tuple[int, int]


class OpBuildError(Exception):
    """A message about why an op cannot be built for its input."""


@dataclass(frozen=True)
class OpDef:
    name: str
    category: str
    params: type[StrictModel]
    build: Callable[[Any, Shape, bool], nn.Module] | None  # (params, input shape, pretrained ok)
    merge: str | None = None  # "add" | "concat" | "multiply"
    description: str = ""

    @property
    def is_merge(self) -> bool:
        return self.merge is not None


OPS: dict[str, OpDef] = {}


def _register(
    name: str, category: str, params: type[StrictModel], description: str, merge: str | None = None
):
    def decorator(build: Callable[[Any, Shape, bool], nn.Module] | None):
        OPS[name] = OpDef(name, category, params, build, merge, description)
        return build

    return decorator


def _image(shape: Shape, op: str) -> int:
    if len(shape) != 3:
        raise OpBuildError(f"{op} needs a C×H×W feature map, but its input is {_fmt(shape)}")
    return shape[0]


def _vector(shape: Shape, op: str) -> int:
    if len(shape) != 1:
        raise OpBuildError(
            f"{op} needs a flat vector, but its input is {_fmt(shape)}; "
            "add Flatten or GlobalAvgPool first"
        )
    return shape[0]


def _fmt(shape: Shape) -> str:
    return "×".join(str(s) for s in shape) if shape else "a scalar"


def _padding(kernel_size: IntOrPair, padding: Any, dilation: IntOrPair = 1) -> Any:
    if padding != "auto":
        return padding
    k = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
    d = dilation if isinstance(dilation, int) else dilation[0]
    return d * (k - 1) // 2


# --------------------------------------------------------------------------- params


class NoParams(StrictModel):
    pass


class ConvParams(StrictModel):
    out_channels: int = Field(ge=1, le=16384)
    kernel_size: IntOrPair = 3
    stride: IntOrPair = 1
    # "auto" keeps H×W for stride 1 (kernel_size // 2).
    padding: Literal["auto"] | IntOrPair = "auto"
    dilation: IntOrPair = 1
    groups: int = Field(default=1, ge=1)
    bias: bool = True


class ConvTransposeParams(StrictModel):
    out_channels: int = Field(ge=1, le=16384)
    kernel_size: IntOrPair = 2
    stride: IntOrPair = 2
    padding: IntOrPair = 0
    output_padding: IntOrPair = 0
    bias: bool = True


class DepthwiseParams(StrictModel):
    kernel_size: int = Field(default=3, ge=1)
    stride: int = Field(default=1, ge=1)
    multiplier: int = Field(default=1, ge=1, le=8)


class SeparableParams(StrictModel):
    out_channels: int = Field(ge=1, le=16384)
    kernel_size: int = Field(default=3, ge=1)
    stride: int = Field(default=1, ge=1)


class GroupNormParams(StrictModel):
    num_groups: int = Field(default=8, ge=1)


class InstanceNormParams(StrictModel):
    affine: bool = True


class LeakyReLUParams(StrictModel):
    negative_slope: float = Field(default=0.01, ge=0, le=1)


class ELUParams(StrictModel):
    alpha: float = Field(default=1.0, gt=0)


class PoolParams(StrictModel):
    kernel_size: IntOrPair = 2
    stride: IntOrPair | None = None
    padding: IntOrPair = 0


class AdaptivePoolParams(StrictModel):
    output_size: IntOrPair = 1


class DropoutParams(StrictModel):
    p: float = Field(default=0.5, ge=0, lt=1)


class LinearParams(StrictModel):
    # Use "$num_outputs" for the task's number of outputs (classes or targets).
    out_features: int = Field(ge=1, le=1_000_000)
    bias: bool = True


class UpsampleParams(StrictModel):
    scale_factor: float = Field(default=2.0, gt=0, le=16)
    mode: Literal["nearest", "bilinear"] = "nearest"


class ConcatParams(StrictModel):
    pass


class BlockParams(StrictModel):
    out_channels: int = Field(ge=1, le=16384)
    stride: int = Field(default=1, ge=1, le=4)


class BottleneckParams(BlockParams):
    expansion: int = Field(default=4, ge=1, le=8)


class DenseBlockParams(StrictModel):
    num_layers: int = Field(default=4, ge=1, le=64)
    growth_rate: int = Field(default=32, ge=1, le=512)
    bn_size: int = Field(default=4, ge=1, le=8)


class SqueezeExciteParams(StrictModel):
    reduction: int = Field(default=16, ge=1, le=64)


class InceptionParams(StrictModel):
    c1x1: int = Field(default=64, ge=1)
    c3x3: int = Field(default=128, ge=1)
    c5x5: int = Field(default=32, ge=1)
    pool_proj: int = Field(default=32, ge=1)


class MBConvParams(StrictModel):
    out_channels: int = Field(ge=1, le=16384)
    expand_ratio: int = Field(default=4, ge=1, le=8)
    kernel_size: Literal[3, 5, 7] = 3
    stride: int = Field(default=1, ge=1, le=2)
    se: bool = True


class BackboneParams(StrictModel):
    source: Literal["timm"] = "timm"
    name: str = Field(min_length=1)
    pretrained: bool = True
    # features: last feature map (C×H×W); pooled: a vector after global pooling.
    output: Literal["features", "pooled"] = "features"


# --------------------------------------------------------------------------- convolution


@_register("Conv2d", "convolution", ConvParams, "2D convolution")
def _conv(p: ConvParams, shape: Shape, pretrained: bool) -> nn.Module:
    in_channels = _image(shape, "Conv2d")
    if in_channels % p.groups or p.out_channels % p.groups:
        raise OpBuildError(
            f"groups={p.groups} must divide both input channels ({in_channels}) "
            f"and out_channels ({p.out_channels})"
        )
    return nn.Conv2d(
        in_channels,
        p.out_channels,
        p.kernel_size,
        p.stride,
        _padding(p.kernel_size, p.padding, p.dilation),
        p.dilation,
        p.groups,
        p.bias,
    )


@_register("ConvTranspose2d", "convolution", ConvTransposeParams, "Learned upsampling")
def _conv_transpose(p: ConvTransposeParams, shape: Shape, pretrained: bool) -> nn.Module:
    return nn.ConvTranspose2d(
        _image(shape, "ConvTranspose2d"),
        p.out_channels,
        p.kernel_size,
        p.stride,
        p.padding,
        p.output_padding,
        bias=p.bias,
    )


@_register("DepthwiseConv2d", "convolution", DepthwiseParams, "One filter per channel")
def _depthwise(p: DepthwiseParams, shape: Shape, pretrained: bool) -> nn.Module:
    channels = _image(shape, "DepthwiseConv2d")
    return nn.Conv2d(
        channels,
        channels * p.multiplier,
        p.kernel_size,
        p.stride,
        p.kernel_size // 2,
        groups=channels,
    )


@_register("SeparableConv2d", "convolution", SeparableParams, "Depthwise + pointwise convolution")
def _separable(p: SeparableParams, shape: Shape, pretrained: bool) -> nn.Module:
    channels = _image(shape, "SeparableConv2d")
    return nn.Sequential(
        nn.Conv2d(
            channels,
            channels,
            p.kernel_size,
            p.stride,
            p.kernel_size // 2,
            groups=channels,
            bias=False,
        ),
        nn.Conv2d(channels, p.out_channels, 1),
    )


# --------------------------------------------------------------------------- normalization


@_register("BatchNorm2d", "normalization", NoParams, "Batch normalization over channels")
def _bn2d(p: NoParams, shape: Shape, pretrained: bool) -> nn.Module:
    return nn.BatchNorm2d(_image(shape, "BatchNorm2d"))


@_register("BatchNorm1d", "normalization", NoParams, "Batch normalization for vectors")
def _bn1d(p: NoParams, shape: Shape, pretrained: bool) -> nn.Module:
    return nn.BatchNorm1d(_vector(shape, "BatchNorm1d"))


@_register("GroupNorm", "normalization", GroupNormParams, "Group normalization")
def _group_norm(p: GroupNormParams, shape: Shape, pretrained: bool) -> nn.Module:
    channels = _image(shape, "GroupNorm")
    if channels % p.num_groups:
        raise OpBuildError(f"num_groups={p.num_groups} must divide the {channels} input channels")
    return nn.GroupNorm(p.num_groups, channels)


@_register(
    "LayerNorm", "normalization", NoParams, "Layer normalization (over channels for feature maps)"
)
def _layer_norm(p: NoParams, shape: Shape, pretrained: bool) -> nn.Module:
    if len(shape) == 3:
        return blocks.LayerNorm2d(shape[0])
    return nn.LayerNorm(_vector(shape, "LayerNorm"))


@_register("InstanceNorm2d", "normalization", InstanceNormParams, "Instance normalization")
def _instance_norm(p: InstanceNormParams, shape: Shape, pretrained: bool) -> nn.Module:
    return nn.InstanceNorm2d(_image(shape, "InstanceNorm2d"), affine=p.affine)


# --------------------------------------------------------------------------- activations

for _name, _factory in {
    "ReLU": lambda p: nn.ReLU(inplace=False),
    "GELU": lambda p: nn.GELU(),
    "SiLU": lambda p: nn.SiLU(),
    "Sigmoid": lambda p: nn.Sigmoid(),
    "Tanh": lambda p: nn.Tanh(),
    "Hardswish": lambda p: nn.Hardswish(),
    "Mish": lambda p: nn.Mish(),
}.items():
    OPS[_name] = OpDef(
        _name,
        "activation",
        NoParams,
        lambda p, s, pre, f=_factory: f(p),
        None,
        f"{_name} activation",
    )

OPS["LeakyReLU"] = OpDef(
    "LeakyReLU",
    "activation",
    LeakyReLUParams,
    lambda p, s, pre: nn.LeakyReLU(p.negative_slope),
    None,
    "Leaky ReLU",
)
OPS["ELU"] = OpDef(
    "ELU", "activation", ELUParams, lambda p, s, pre: nn.ELU(p.alpha), None, "ELU activation"
)


# --------------------------------------------------------------------------- pooling


@_register("MaxPool2d", "pooling", PoolParams, "Max pooling")
def _max_pool(p: PoolParams, shape: Shape, pretrained: bool) -> nn.Module:
    _image(shape, "MaxPool2d")
    return nn.MaxPool2d(p.kernel_size, p.stride, p.padding)


@_register("AvgPool2d", "pooling", PoolParams, "Average pooling")
def _avg_pool(p: PoolParams, shape: Shape, pretrained: bool) -> nn.Module:
    _image(shape, "AvgPool2d")
    return nn.AvgPool2d(p.kernel_size, p.stride, p.padding)


@_register("AdaptiveAvgPool2d", "pooling", AdaptivePoolParams, "Average pooling to a fixed size")
def _adaptive_avg(p: AdaptivePoolParams, shape: Shape, pretrained: bool) -> nn.Module:
    _image(shape, "AdaptiveAvgPool2d")
    return nn.AdaptiveAvgPool2d(p.output_size)


@_register("AdaptiveMaxPool2d", "pooling", AdaptivePoolParams, "Max pooling to a fixed size")
def _adaptive_max(p: AdaptivePoolParams, shape: Shape, pretrained: bool) -> nn.Module:
    _image(shape, "AdaptiveMaxPool2d")
    return nn.AdaptiveMaxPool2d(p.output_size)


@_register("GlobalAvgPool", "pooling", NoParams, "Average each channel to one value (C×H×W → C)")
def _global_avg(p: NoParams, shape: Shape, pretrained: bool) -> nn.Module:
    _image(shape, "GlobalAvgPool")
    return nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())


# --------------------------------------------------------------------------- regularization / shape


@_register("Dropout", "regularization", DropoutParams, "Randomly zero values while training")
def _dropout(p: DropoutParams, shape: Shape, pretrained: bool) -> nn.Module:
    return nn.Dropout(p.p)


@_register(
    "Dropout2d", "regularization", DropoutParams, "Randomly zero whole channels while training"
)
def _dropout2d(p: DropoutParams, shape: Shape, pretrained: bool) -> nn.Module:
    _image(shape, "Dropout2d")
    return nn.Dropout2d(p.p)


@_register("Flatten", "shape", NoParams, "Flatten a feature map into a vector")
def _flatten(p: NoParams, shape: Shape, pretrained: bool) -> nn.Module:
    return nn.Flatten()


@_register("Upsample", "shape", UpsampleParams, "Resize a feature map")
def _upsample(p: UpsampleParams, shape: Shape, pretrained: bool) -> nn.Module:
    _image(shape, "Upsample")
    return nn.Upsample(scale_factor=p.scale_factor, mode=p.mode)


@_register("Identity", "shape", NoParams, "Pass the input through unchanged")
def _identity(p: NoParams, shape: Shape, pretrained: bool) -> nn.Module:
    return nn.Identity()


@_register("Linear", "linear", LinearParams, "Fully connected layer")
def _linear(p: LinearParams, shape: Shape, pretrained: bool) -> nn.Module:
    return nn.Linear(_vector(shape, "Linear"), p.out_features, p.bias)


# --------------------------------------------------------------------------- merges

_register("Add", "merge", NoParams, "Element-wise sum (residual connection)", merge="add")(None)
_register("Concat", "merge", ConcatParams, "Join inputs along the channel axis", merge="concat")(
    None
)
_register("Multiply", "merge", NoParams, "Element-wise product (gating)", merge="multiply")(None)


# --------------------------------------------------------------------------- blocks


@_register("BasicBlock", "block", BlockParams, "ResNet basic residual block")
def _basic_block(p: BlockParams, shape: Shape, pretrained: bool) -> nn.Module:
    return blocks.BasicBlock(_image(shape, "BasicBlock"), p.out_channels, p.stride)


@_register("Bottleneck", "block", BottleneckParams, "ResNet bottleneck residual block")
def _bottleneck(p: BottleneckParams, shape: Shape, pretrained: bool) -> nn.Module:
    return blocks.Bottleneck(_image(shape, "Bottleneck"), p.out_channels, p.stride, p.expansion)


@_register(
    "DenseBlock", "block", DenseBlockParams, "DenseNet block (each layer sees all previous ones)"
)
def _dense_block(p: DenseBlockParams, shape: Shape, pretrained: bool) -> nn.Module:
    return blocks.DenseBlock(_image(shape, "DenseBlock"), p.num_layers, p.growth_rate, p.bn_size)


@_register(
    "SqueezeExcite", "block", SqueezeExciteParams, "Channel attention (squeeze-and-excitation)"
)
def _se(p: SqueezeExciteParams, shape: Shape, pretrained: bool) -> nn.Module:
    return blocks.SqueezeExcite(_image(shape, "SqueezeExcite"), p.reduction)


@_register(
    "InceptionBlock", "block", InceptionParams, "Parallel 1×1, 3×3, 5×5 and pooling branches"
)
def _inception(p: InceptionParams, shape: Shape, pretrained: bool) -> nn.Module:
    return blocks.InceptionBlock(
        _image(shape, "InceptionBlock"), p.c1x1, p.c3x3, p.c5x5, p.pool_proj
    )


@_register("MBConv", "block", MBConvParams, "MobileNet/EfficientNet inverted residual block")
def _mbconv(p: MBConvParams, shape: Shape, pretrained: bool) -> nn.Module:
    return blocks.MBConv(
        _image(shape, "MBConv"), p.out_channels, p.expand_ratio, p.kernel_size, p.stride, p.se
    )


@_register(
    "Backbone", "backbone", BackboneParams, "A whole zoo model (pretrained or blank) as one block"
)
def _backbone(p: BackboneParams, shape: Shape, pretrained: bool) -> nn.Module:
    channels = _image(shape, "Backbone")
    return blocks.TimmBackbone(p.name, channels, p.pretrained and pretrained, p.output == "pooled")


# --------------------------------------------------------------------------- helpers


def resolve_symbols(params: dict[str, Any], symbols: dict[str, int]) -> dict[str, Any]:
    """Replace "$num_outputs" style values with numbers."""
    resolved = {}
    for key, value in params.items():
        if isinstance(value, str) and value.startswith("$"):
            name = value[1:]
            if name not in symbols:
                raise OpBuildError(
                    f"unknown symbol {value} (known: {', '.join('$' + s for s in symbols)})"
                )
            resolved[key] = symbols[name]
        else:
            resolved[key] = value
    return resolved


def parse_params(op: OpDef, params: dict[str, Any], symbols: dict[str, int]) -> Any:
    """Validated params; raises pydantic.ValidationError or OpBuildError."""
    return op.params.model_validate(resolve_symbols(params, symbols))


def palette() -> list[dict[str, Any]]:
    return [
        {
            "op": op.name,
            "category": op.category,
            "description": op.description,
            "merge": op.is_merge,
            "params": op.params.model_json_schema(),
        }
        for op in OPS.values()
    ]
