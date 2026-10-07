"""Architecture specs: a pretrained base plus patches, or a full layer graph (PLAN §9)."""

from typing import Annotated, Any, Literal

from pydantic import Field

from netpattern_engine.spec.base import StrictModel


class OpSpec(StrictModel):
    """A layer from the op palette, e.g. {"op": "Linear", "params": {"out_features": 10}}."""

    op: str = Field(min_length=1)
    params: dict[str, Any] = {}


class PretrainedBase(StrictModel):
    source: Literal["torchvision", "timm", "netpattern"]
    # torchvision: resnet18, deeplabv3_resnet50, fasterrcnn_resnet50_fpn_v2, ...
    # timm: resnet50.a1_in1k, convnext_tiny, ...
    # netpattern: unet (encoder taken from timm)
    name: str = Field(min_length=1)
    # "DEFAULT", a specific weights name (e.g. IMAGENET1K_V1), or null for a blank model.
    weights: str | None = "DEFAULT"
    # U-Net encoder (a timm model name).
    encoder: str | None = None
    # Family-specific settings, e.g. {"min_size": 512} for detection models.
    options: dict[str, Any] = {}


class FreezePatch(StrictModel):
    op: Literal["freeze"]
    # Qualified module names, or @all / @backbone (everything but the task head) / @head.
    targets: list[str] = Field(min_length=1)


class UnfreezePatch(StrictModel):
    op: Literal["unfreeze"]
    targets: list[str] = Field(min_length=1)


class ReplacePatch(StrictModel):
    op: Literal["replace"]
    target: str = Field(min_length=1)
    with_: list[OpSpec] = Field(alias="with", min_length=1)


class InsertAfterPatch(StrictModel):
    op: Literal["insert_after"]
    target: str = Field(min_length=1)
    ops: list[OpSpec] = Field(min_length=1)


class RemovePatch(StrictModel):
    """Replaces a module with Identity."""

    op: Literal["remove"]
    target: str = Field(min_length=1)


Patch = Annotated[
    FreezePatch | UnfreezePatch | ReplacePatch | InsertAfterPatch | RemovePatch,
    Field(discriminator="op"),
]


class PretrainedArchitecture(StrictModel):
    kind: Literal["pretrained"]
    base: PretrainedBase
    # Replace the classifier/predictor with one sized for the task (PLAN §7.4).
    replace_head: bool = True
    patches: list[Patch] = []


class GraphInput(StrictModel):
    # Channels, height, width.
    shape: tuple[int, int, int]


class GraphNode(StrictModel):
    id: str = Field(min_length=1, max_length=64)
    op: str = Field(min_length=1)
    params: dict[str, Any] = {}


class GraphArchitecture(StrictModel):
    """A custom CNN drawn in the layer editor. `in_*` sizes are inferred, never typed."""

    kind: Literal["graph"]
    input: GraphInput
    nodes: list[GraphNode] = Field(min_length=1)
    # (from, to) pairs; "input" is the implicit input node.
    edges: list[tuple[str, str]] = Field(min_length=1)
    # The output node; defaults to the only node without outgoing edges.
    output: str | None = None


ArchitectureSpec = Annotated[
    PretrainedArchitecture | GraphArchitecture,
    Field(discriminator="kind"),
]
