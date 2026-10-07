from typing import Any, Literal

from pydantic import Field

from netpattern_engine.spec.base import StrictModel


class TransformOp(StrictModel):
    """One augmentation step; `op` must be in the engine's transform palette."""

    op: str = Field(min_length=1)
    params: dict[str, Any] = {}


class TransformsSpec(StrictModel):
    """Preprocessing and augmentation (PLAN §7.3).

    from_weights: input size and normalization come from the pretrained weights.
    basic: resize to the model's input size, normalize to [-1, 1].
    none: only the listed ops (the user takes care of size and normalization).
    """

    preset: Literal["from_weights", "basic", "none"] = "from_weights"
    # Overrides the input size the preset would use (square side length in pixels).
    image_size: int | None = Field(default=None, ge=16, le=2048)
    train: list[TransformOp] = []
    eval: list[TransformOp] = []
