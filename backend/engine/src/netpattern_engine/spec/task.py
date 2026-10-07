from typing import Literal

from pydantic import Field

from netpattern_engine.spec.base import StrictModel

TaskType = Literal[
    "classification.single_label",
    "classification.multi_label",
    "regression",
    "segmentation.semantic",
    "detection.bbox",
]

TASK_TYPES: tuple[str, ...] = TaskType.__args__  # type: ignore[attr-defined]


class TaskSpec(StrictModel):
    """What the user wants to predict (PLAN §6)."""

    type: TaskType
    # Class names in index order. Omitted: taken from the dataset.
    # Segmentation: index 0 is usually "background". Detection: object classes only.
    classes: list[str] | None = Field(default=None, min_length=1)
    # Regression: names of the predicted values. Omitted: taken from the dataset.
    targets: list[str] | None = Field(default=None, min_length=1)
    # Segmentation: mask value that is ignored by the loss and metrics.
    ignore_index: int = 255
    # Multi-label: probability above which a label counts as predicted.
    threshold: float = Field(default=0.5, gt=0, lt=1)
