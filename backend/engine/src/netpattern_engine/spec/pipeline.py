"""Pipeline spec (the canvas) and its resolution into one typed object for the engine."""

from typing import Any, Literal

from pydantic import Field, TypeAdapter, ValidationError

from netpattern_engine.errors import Issue, SpecError, issues_from_validation_error
from netpattern_engine.spec.architecture import ArchitectureSpec
from netpattern_engine.spec.base import StrictModel
from netpattern_engine.spec.data import DATASET_FORMATS, DatasetSpec
from netpattern_engine.spec.task import TaskSpec
from netpattern_engine.spec.training import TrainingSpec
from netpattern_engine.spec.transforms import TransformsSpec

SPEC_VERSION = "0.1"

NodeType = Literal["dataset", "transforms", "model", "trainer", "evaluator"]

# The canvas is a chain in this order; transforms and evaluator are optional.
CHAIN_ORDER: tuple[str, ...] = ("dataset", "transforms", "model", "trainer", "evaluator")
REQUIRED_NODES = ("dataset", "model", "trainer")


class PipelineNode(StrictModel):
    id: str = Field(min_length=1, max_length=64)
    type: NodeType
    params: dict[str, Any] = {}


class PipelineSpec(StrictModel):
    spec_version: Literal["0.1"] = SPEC_VERSION
    task: TaskSpec
    nodes: list[PipelineNode] = Field(min_length=1)
    edges: list[tuple[str, str]] = []


class EvaluationSpec(StrictModel):
    splits: list[Literal["val", "test"]] = ["val", "test"]


class ModelNodeParams(StrictModel):
    architecture: ArchitectureSpec


class ResolvedPipeline(StrictModel):
    """Everything the engine needs, typed and validated."""

    task: TaskSpec
    data: DatasetSpec
    transforms: TransformsSpec = TransformsSpec()
    architecture: ArchitectureSpec
    training: TrainingSpec = TrainingSpec()
    evaluation: EvaluationSpec = EvaluationSpec()


_DATASET_ADAPTER: TypeAdapter[Any] = TypeAdapter(DatasetSpec)


def resolve_pipeline(raw: dict[str, Any] | PipelineSpec) -> ResolvedPipeline:
    """Validate the canvas and turn it into a ResolvedPipeline, or raise SpecError."""
    try:
        spec = raw if isinstance(raw, PipelineSpec) else PipelineSpec.model_validate(raw)
    except ValidationError as error:
        raise SpecError(issues_from_validation_error(error)) from error

    issues: list[Issue] = []
    by_type: dict[str, PipelineNode] = {}
    ids = set()
    for node in spec.nodes:
        if node.id in ids:
            issues.append(Issue("duplicate node id", node_id=node.id))
        ids.add(node.id)
        if node.type in by_type:
            issues.append(
                Issue(f"only one '{node.type}' node is allowed", node_id=node.id, field="type")
            )
        by_type.setdefault(node.type, node)

    for required in REQUIRED_NODES:
        if required not in by_type:
            issues.append(Issue(f"the pipeline needs a '{required}' node"))

    issues.extend(_check_edges(spec, by_type))

    parsed: dict[str, Any] = {}
    parsers: dict[str, Any] = {
        "dataset": lambda params: _DATASET_ADAPTER.validate_python(params),
        "transforms": TransformsSpec.model_validate,
        "model": lambda params: ModelNodeParams.model_validate(params).architecture,
        "trainer": TrainingSpec.model_validate,
        "evaluator": EvaluationSpec.model_validate,
    }
    for node_type, node in by_type.items():
        try:
            parsed[node_type] = parsers[node_type](node.params)
        except ValidationError as error:
            issues.extend(issues_from_validation_error(error, node_id=node.id, prefix="params"))

    dataset = parsed.get("dataset")
    if dataset is not None and spec.task.type not in DATASET_FORMATS[dataset.format]:
        issues.append(
            Issue(
                f"dataset format '{dataset.format}' does not support task '{spec.task.type}'",
                node_id=by_type["dataset"].id,
                field="params.format",
            )
        )

    if issues:
        raise SpecError(issues)

    return ResolvedPipeline(
        task=spec.task,
        data=parsed["dataset"],
        transforms=parsed.get("transforms", TransformsSpec()),
        architecture=parsed["model"],
        training=parsed["trainer"],
        evaluation=parsed.get("evaluator", EvaluationSpec()),
    )


def _check_edges(spec: PipelineSpec, by_type: dict[str, PipelineNode]) -> list[Issue]:
    """Edges must follow the chain dataset -> transforms -> model -> trainer -> evaluator."""
    if not spec.edges:
        return []  # an edge-less spec is read in chain order

    issues = []
    node_types = {node.id: node.type for node in spec.nodes}
    present = [node_type for node_type in CHAIN_ORDER if node_type in by_type]
    expected = {(by_type[a].id, by_type[b].id) for a, b in zip(present, present[1:])}
    for source, target in spec.edges:
        for end in (source, target):
            if end not in node_types:
                issues.append(Issue(f"edge refers to unknown node '{end}'"))
        if (source, target) not in expected and source in node_types and target in node_types:
            issues.append(
                Issue(
                    f"a {node_types[source]} node cannot connect to a {node_types[target]} node",
                    node_id=target,
                )
            )
    for source, target in sorted(expected - set(spec.edges)):
        issues.append(Issue(f"missing connection from '{source}'", node_id=target))
    return issues
