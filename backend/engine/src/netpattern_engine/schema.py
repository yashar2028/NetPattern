"""The engine's JSON Schema and capabilities, extracted when an environment is built.

The UI renders node forms from this document (PLAN P6), so every sandbox shows
exactly the options its own pinned engine supports.
"""

from __future__ import annotations

from typing import Any

from pydantic import TypeAdapter

from netpattern_engine import __version__
from netpattern_engine.spec import (
    SPEC_VERSION,
    ArchitectureSpec,
    DatasetSpec,
    PipelineSpec,
    TaskSpec,
)
from netpattern_engine.spec.data import DATASET_FORMATS
from netpattern_engine.spec.training import TrainingSpec
from netpattern_engine.spec.transforms import TransformsSpec


def schema_document() -> dict[str, Any]:
    from netpattern_engine.data.transforms import transform_palette
    from netpattern_engine.graph.ops import palette
    from netpattern_engine.tasks import TASKS

    return {
        "engine": __version__,
        "spec_version": SPEC_VERSION,
        "schemas": {
            "pipeline": PipelineSpec.model_json_schema(),
            "task": TaskSpec.model_json_schema(),
            "dataset": TypeAdapter(DatasetSpec).json_schema(),
            "transforms": TransformsSpec.model_json_schema(),
            "architecture": TypeAdapter(ArchitectureSpec).json_schema(by_alias=True),
            "training": TrainingSpec.model_json_schema(),
        },
        "capabilities": {
            "tasks": [
                {
                    "type": name,
                    "monitor": plugin.monitor,
                    "custom_graphs": plugin.supports_graph,
                    "packs": list(plugin.required_packs),
                }
                for name, plugin in TASKS.items()
            ],
            "dataset_formats": {name: list(tasks) for name, tasks in DATASET_FORMATS.items()},
            "graph_ops": palette(),
            "transform_ops": transform_palette(),
            "packs": {
                "medical": "DICOM and NIfTI images and masks",
                "detection": "object detection metrics (mAP)",
            },
        },
    }
