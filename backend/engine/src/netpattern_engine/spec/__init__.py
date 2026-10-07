"""Versioned JSON spec: the single contract between UI, backend and engine (PLAN §9).

This package must not import torch, so the platform can parse specs quickly.
"""

from netpattern_engine.spec.architecture import (
    ArchitectureSpec,
    GraphArchitecture,
    GraphNode,
    OpSpec,
    PretrainedArchitecture,
    PretrainedBase,
)
from netpattern_engine.spec.data import DatasetSpec, IntensitySpec, SliceSpec, SplitSpec
from netpattern_engine.spec.pipeline import (
    SPEC_VERSION,
    EvaluationSpec,
    PipelineNode,
    PipelineSpec,
    ResolvedPipeline,
    resolve_pipeline,
)
from netpattern_engine.spec.task import TASK_TYPES, TaskSpec
from netpattern_engine.spec.training import TrainingSpec
from netpattern_engine.spec.transforms import TransformOp, TransformsSpec

__all__ = [
    "SPEC_VERSION",
    "TASK_TYPES",
    "ArchitectureSpec",
    "DatasetSpec",
    "EvaluationSpec",
    "GraphArchitecture",
    "GraphNode",
    "IntensitySpec",
    "OpSpec",
    "PipelineNode",
    "PipelineSpec",
    "PretrainedArchitecture",
    "PretrainedBase",
    "ResolvedPipeline",
    "SliceSpec",
    "SplitSpec",
    "TaskSpec",
    "TrainingSpec",
    "TransformOp",
    "TransformsSpec",
    "resolve_pipeline",
]
