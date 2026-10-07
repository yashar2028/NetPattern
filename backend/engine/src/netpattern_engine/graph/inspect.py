"""Show a (pretrained) model as a graph for the layer editor (PLAN §7.5).

The model is traced with torch.fx and shapes are propagated on the meta device.
Nodes carry their module path so the UI can collapse layer1 … layer4 into groups.
Models that cannot be traced (e.g. detection models) fall back to a module tree;
those are still editable through patches by qualified name.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import fx, nn


def module_tree(model: nn.Module, max_depth: int = 3) -> list[dict[str, Any]]:
    entries = []
    for name, module in model.named_modules():
        if not name or name.count(".") >= max_depth:
            continue
        entries.append(
            {
                "name": name,
                "type": type(module).__name__,
                "params": sum(p.numel() for p in module.parameters()),
                "trainable": any(p.requires_grad for p in module.parameters()),
                "leaf": not any(True for _ in module.children()),
            }
        )
    return entries


def inspect_model(model: nn.Module, input_shape: tuple[int, ...]) -> dict[str, Any]:
    tree = module_tree(model)
    try:
        traced = fx.symbolic_trace(model)
    except Exception as error:  # dynamic control flow, e.g. detection models
        return {"traceable": False, "reason": str(error).splitlines()[0][:200], "modules": tree}

    shapes: dict[str, list[int]] = {}
    try:
        import copy

        from torch.fx.passes.shape_prop import ShapeProp

        # The traced module shares parameters with `model`; never move real weights.
        on_meta = all(p.device.type == "meta" for p in traced.parameters())
        meta = (traced if on_meta else copy.deepcopy(traced)).to("meta").eval()
        ShapeProp(meta).propagate(torch.empty((1, *input_shape), device="meta"))
        for node in meta.graph.nodes:
            tensor_meta = node.meta.get("tensor_meta")
            if tensor_meta is not None and hasattr(tensor_meta, "shape"):
                shapes[node.name] = list(tensor_meta.shape[1:])
    except Exception:
        meta = traced

    named = dict(model.named_modules())
    nodes, edges = [], []
    for node in meta.graph.nodes:
        entry: dict[str, Any] = {"id": node.name, "kind": node.op, "shape": shapes.get(node.name)}
        if node.op == "call_module":
            module = named.get(str(node.target))
            entry.update(
                target=str(node.target),
                type=type(module).__name__ if module is not None else None,
                group=str(node.target).rsplit(".", 1)[0] if "." in str(node.target) else None,
                params=(
                    sum(p.numel() for p in module.parameters(recurse=False))
                    if module is not None
                    else 0
                ),
            )
        elif node.op in ("call_function", "call_method"):
            entry["target"] = getattr(node.target, "__name__", str(node.target))
        nodes.append(entry)
        for source in node.all_input_nodes:
            edges.append([source.name, node.name])
    return {"traceable": True, "nodes": nodes, "edges": edges, "modules": tree}
