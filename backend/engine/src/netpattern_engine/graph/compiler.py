"""Custom graph -> torch.fx.GraphModule, with meta-device shape inference (PLAN §9).

Shapes are computed by building each op on PyTorch's `meta` device and running a
forward pass: exact for every op, no memory allocated, so the editor can call it on
every change. Problems are reported per node so the UI can highlight them.
"""

from __future__ import annotations

import operator
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

import torch
from pydantic import ValidationError
from torch import fx, nn

from netpattern_engine.errors import Issue, SpecError, issues_from_validation_error
from netpattern_engine.graph.ops import OPS, OpBuildError, OpDef, Shape, parse_params
from netpattern_engine.spec.architecture import GraphArchitecture, OpSpec

INPUT = "input"
META_BATCH = 2


@dataclass
class GraphResult:
    shapes: dict[str, Shape] = field(default_factory=dict)  # node id -> output shape (no batch)
    params: dict[str, int] = field(default_factory=dict)  # node id -> parameter count
    issues: list[Issue] = field(default_factory=list)
    output: str | None = None
    module: nn.Module | None = None  # set when materialized
    flops: int | None = None

    @property
    def output_shape(self) -> Shape | None:
        return self.shapes.get(self.output) if self.output else None

    @property
    def total_params(self) -> int:
        return sum(self.params.values())


# --------------------------------------------------------------------------- structure


def _structure(
    arch: GraphArchitecture,
) -> tuple[list[str], dict[str, list[str]], str | None, list[Issue]]:
    """Topological order, ordered predecessors per node, output node, issues."""
    issues: list[Issue] = []
    nodes = {node.id: node for node in arch.nodes}
    if INPUT in nodes:
        issues.append(Issue("'input' is reserved for the graph input", node_id=INPUT))
    if len(nodes) != len(arch.nodes):
        seen: set[str] = set()
        for node in arch.nodes:
            if node.id in seen:
                issues.append(Issue("duplicate node id", node_id=node.id))
            seen.add(node.id)

    predecessors: dict[str, list[str]] = defaultdict(list)
    successors: dict[str, list[str]] = defaultdict(list)
    for source, target in arch.edges:
        if source != INPUT and source not in nodes:
            issues.append(Issue(f"edge comes from unknown node '{source}'", node_id=target))
            continue
        if target not in nodes:
            issues.append(Issue(f"edge goes to unknown node '{target}'", node_id=source))
            continue
        if source == target:
            issues.append(Issue("a node cannot connect to itself", node_id=target))
            continue
        predecessors[target].append(source)
        successors[source].append(target)

    for node in arch.nodes:
        op = OPS.get(node.op)
        count = len(predecessors[node.id])
        if op is None:
            issues.append(Issue(f"unknown op '{node.op}'", node_id=node.id, field="op"))
        elif op.is_merge and count < 2:
            issues.append(
                Issue(f"{node.op} needs at least two inputs, has {count}", node_id=node.id)
            )
        elif not op.is_merge and count != 1:
            issues.append(Issue(f"{node.op} needs exactly one input, has {count}", node_id=node.id))

    # Kahn's algorithm starting at the input; nodes in cycles or detached never get queued.
    indegree = {node_id: len(predecessors[node_id]) for node_id in nodes}
    queue = deque([INPUT])
    order: list[str] = []
    visited: set[str] = set()
    while queue:
        current = queue.popleft()
        for successor in successors[current]:
            indegree[successor] -= 1
            if indegree[successor] == 0 and successor not in visited:
                visited.add(successor)
                order.append(successor)
                queue.append(successor)
    if not successors[INPUT]:
        issues.append(Issue("nothing is connected to the input"))
    for node_id in nodes:
        if node_id not in visited and successors[INPUT]:
            issues.append(
                Issue("not reachable from the input, or part of a cycle", node_id=node_id)
            )

    if arch.output is not None:
        output = arch.output if arch.output in nodes else None
        if output is None:
            issues.append(Issue(f"output node '{arch.output}' does not exist", field="output"))
    else:
        sinks = [node_id for node_id in order if not successors[node_id]]
        output = sinks[0] if len(sinks) == 1 else None
        if len(sinks) > 1:
            issues.append(
                Issue(
                    f"the graph has several ends ({', '.join(sinks)}); choose one as output",
                    field="output",
                )
            )
    return order, predecessors, output, issues


# --------------------------------------------------------------------------- shapes


def _merge_shape(op: OpDef, shapes: list[Shape]) -> Shape:
    first = shapes[0]
    if op.merge == "add":
        for shape in shapes[1:]:
            if shape != first:
                raise OpBuildError(f"shape mismatch at Add: {_fmt(first)} vs {_fmt(shape)}")
        return first
    if op.merge == "multiply":
        a = torch.empty((META_BATCH, *first), device="meta")
        try:
            for shape in shapes[1:]:
                a = a * torch.empty((META_BATCH, *shape), device="meta")
        except RuntimeError:
            raise OpBuildError(
                "Multiply inputs must have the same shape or broadcast (e.g. C×1×1 with C×H×W): "
                + " vs ".join(_fmt(s) for s in shapes)
            ) from None
        return tuple(a.shape[1:])
    # concat along channels
    if any(len(shape) != len(first) for shape in shapes) or any(
        shape[1:] != first[1:] for shape in shapes
    ):
        raise OpBuildError(
            "Concat needs inputs with the same size apart from channels: "
            + " vs ".join(_fmt(s) for s in shapes)
        )
    return (sum(shape[0] for shape in shapes), *first[1:])


def _fmt(shape: Shape) -> str:
    return "×".join(str(s) for s in shape)


def _meta_output(module: nn.Module, shape: Shape) -> Shape:
    module.eval()
    with torch.no_grad():
        out = module(torch.empty((META_BATCH, *shape), device="meta"))
    if not isinstance(out, torch.Tensor):
        raise OpBuildError("op returned more than one tensor")
    return tuple(out.shape[1:])


def build_op(
    op: OpDef, params: Any, shape: Shape, materialize: bool, pretrained: bool
) -> nn.Module:
    if materialize:
        return op.build(params, shape, pretrained)
    from netpattern_engine.meta import meta_init

    with meta_init():
        return op.build(params, shape, False)


def _clean(message: str) -> str:
    message = str(message).strip().splitlines()[0] if str(message).strip() else "failed"
    return re.sub(r"\s+", " ", message)[:300]


def analyze_graph(
    arch: GraphArchitecture,
    symbols: dict[str, int],
    materialize: bool = False,
    pretrained: bool = True,
) -> GraphResult:
    """Shapes, parameter counts and issues; with `materialize`, also the real module."""
    order, predecessors, output, issues = _structure(arch)
    result = GraphResult(issues=issues, output=output)
    nodes = {node.id: node for node in arch.nodes}
    result.shapes[INPUT] = tuple(arch.input.shape)
    modules: dict[str, nn.Module] = {}
    failed: set[str] = set()

    for node_id in order:
        node = nodes[node_id]
        op = OPS.get(node.op)
        inputs = predecessors[node_id]
        if op is None or any(p in failed or p not in result.shapes for p in inputs):
            failed.add(node_id)
            continue
        in_shapes = [result.shapes[p] for p in inputs]
        try:
            params = parse_params(op, node.params, symbols)
            if op.is_merge:
                result.shapes[node_id] = _merge_shape(op, in_shapes)
                result.params[node_id] = 0
                continue
            meta_module = build_op(op, params, in_shapes[0], materialize=False, pretrained=False)
            result.shapes[node_id] = _meta_output(meta_module, in_shapes[0])
            result.params[node_id] = sum(p.numel() for p in meta_module.parameters())
            modules[node_id] = (
                build_op(op, params, in_shapes[0], materialize=True, pretrained=pretrained)
                if materialize
                else meta_module
            )
        except ValidationError as error:
            result.issues.extend(
                issues_from_validation_error(error, node_id=node_id, prefix="params")
            )
            failed.add(node_id)
        except (OpBuildError, RuntimeError, ValueError, TypeError) as error:
            result.issues.append(Issue(_clean(str(error)), node_id=node_id))
            failed.add(node_id)

    if result.issues or output is None:
        return result

    result.module = _to_fx(arch, order, predecessors, output, modules)
    if not materialize:
        result.flops = count_flops(result.module, tuple(arch.input.shape))
    return result


def compile_graph(
    arch: GraphArchitecture, symbols: dict[str, int], pretrained: bool = True
) -> GraphResult:
    result = analyze_graph(arch, symbols, materialize=True, pretrained=pretrained)
    if result.issues:
        raise SpecError(result.issues)
    return result


def _safe_names(node_ids: list[str]) -> dict[str, str]:
    names: dict[str, str] = {}
    used: set[str] = set()
    for node_id in node_ids:
        base = re.sub(r"\W", "_", node_id)
        if not base or base[0].isdigit():
            base = f"n_{base}"
        name, i = base, 1
        while name in used:
            name, i = f"{base}_{i}", i + 1
        used.add(name)
        names[node_id] = name
    return names


def _to_fx(
    arch: GraphArchitecture,
    order: list[str],
    predecessors: dict[str, list[str]],
    output: str,
    modules: dict[str, nn.Module],
) -> fx.GraphModule:
    nodes = {node.id: node for node in arch.nodes}
    names = _safe_names(order)
    root = nn.Module()
    graph = fx.Graph()
    values: dict[str, fx.Node] = {INPUT: graph.placeholder("x")}
    for node_id in order:
        op = OPS[nodes[node_id].op]
        args = [values[p] for p in predecessors[node_id]]
        if op.merge == "add":
            value = args[0]
            for arg in args[1:]:
                value = graph.call_function(operator.add, (value, arg))
        elif op.merge == "multiply":
            value = args[0]
            for arg in args[1:]:
                value = graph.call_function(operator.mul, (value, arg))
        elif op.merge == "concat":
            value = graph.call_function(torch.cat, (tuple(args),), {"dim": 1})
        else:
            root.add_module(names[node_id], modules[node_id])
            value = graph.call_module(names[node_id], (args[0],))
        values[node_id] = value
    graph.output(values[output])
    return fx.GraphModule(root, graph, class_name="NetPatternGraph")


def count_flops(module: nn.Module, input_shape: Shape) -> int | None:
    """FLOPs for one image, counted on the meta device."""
    from torch.utils.flop_counter import FlopCounterMode

    try:
        import copy

        # Never move a real model's weights: count on a meta copy instead.
        on_meta = all(p.device.type == "meta" for p in module.parameters())
        meta = (module if on_meta else copy.deepcopy(module)).to("meta").eval()
        with FlopCounterMode(display=False) as counter, torch.no_grad():
            meta(torch.empty((1, *input_shape), device="meta"))
        return int(counter.get_total_flops())
    except Exception:
        return None


def build_chain(
    ops: list[OpSpec],
    in_shape: Shape,
    symbols: dict[str, int],
    materialize: bool = True,
    pretrained: bool = True,
    field_prefix: str = "",
) -> tuple[nn.Sequential, Shape]:
    """A linear chain of palette ops (used by patches), sized to its input."""
    layers: list[nn.Module] = []
    shape = in_shape
    issues: list[Issue] = []
    for i, spec in enumerate(ops):
        op = OPS.get(spec.op)
        where = f"{field_prefix}[{i}]"
        if op is None or op.is_merge:
            issues.append(Issue(f"'{spec.op}' cannot be used here", field=f"{where}.op"))
            break
        try:
            params = parse_params(op, spec.params, symbols)
            meta_module = build_op(op, params, shape, materialize=False, pretrained=False)
            next_shape = _meta_output(meta_module, shape)
            layers.append(
                build_op(op, params, shape, materialize, pretrained) if materialize else meta_module
            )
            shape = next_shape
        except ValidationError as error:
            issues.extend(issues_from_validation_error(error, prefix=f"{where}.params"))
            break
        except (OpBuildError, RuntimeError, ValueError, TypeError) as error:
            issues.append(Issue(_clean(str(error)), field=where))
            break
    if issues:
        raise SpecError(issues)
    return nn.Sequential(*layers), shape
