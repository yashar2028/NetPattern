"""Apply architecture patches to a built model by qualified module name (PLAN §9)."""

from __future__ import annotations

import copy

import torch
from torch import nn

from netpattern_engine.errors import Issue, SpecError
from netpattern_engine.graph.compiler import build_chain
from netpattern_engine.models.heads import set_submodule
from netpattern_engine.spec.architecture import (
    FreezePatch,
    InsertAfterPatch,
    RemovePatch,
    ReplacePatch,
    UnfreezePatch,
)

META_BATCH = 2


def capture_shapes(
    model: nn.Module, input_shape: tuple[int, ...], targets: list[str]
) -> dict[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    """(input shape, output shape) of the target modules, from one meta forward pass."""
    on_meta = all(p.device.type == "meta" for p in model.parameters())
    meta = (model if on_meta else copy.deepcopy(model)).to("meta").eval()
    shapes: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {}
    hooks = []
    for target in targets:
        module = meta.get_submodule(target)

        def hook(module, inputs, output, target=target):
            if isinstance(inputs[0], torch.Tensor) and isinstance(output, torch.Tensor):
                shapes[target] = (tuple(inputs[0].shape[1:]), tuple(output.shape[1:]))

        hooks.append(module.register_forward_hook(hook))
    try:
        with torch.no_grad():
            meta(torch.empty((META_BATCH, *input_shape), device="meta"))
    finally:
        for handle in hooks:
            handle.remove()
    return shapes


def _module_exists(model: nn.Module, name: str) -> bool:
    try:
        model.get_submodule(name)
        return True
    except AttributeError:
        return False


def _parameters_for(model: nn.Module, target: str, head: list[str]) -> list[nn.Parameter]:
    named = list(model.named_parameters())

    def under(name: str, prefixes: list[str]) -> bool:
        return any(name == p or name.startswith(p + ".") for p in prefixes)

    if target == "@all":
        return [p for _, p in named]
    if target == "@head":
        return [p for n, p in named if under(n, head)]
    if target == "@backbone":
        return [p for n, p in named if not under(n, head)]
    return [p for n, p in named if under(n, [target])]


def apply_patches(
    model: nn.Module,
    patches: list,
    head: list[str],
    input_shape: tuple[int, ...],
    symbols: dict[str, int],
    materialize: bool,
) -> None:
    issues: list[Issue] = []
    for i, patch in enumerate(patches):
        where = f"architecture.patches[{i}]"
        if isinstance(patch, (FreezePatch, UnfreezePatch)):
            for target in patch.targets:
                if not target.startswith("@") and not _module_exists(model, target):
                    issues.append(Issue(f"no module named '{target}'", field=f"{where}.targets"))
                    continue
                for parameter in _parameters_for(model, target, head):
                    parameter.requires_grad = isinstance(patch, UnfreezePatch)
            continue

        if not _module_exists(model, patch.target):
            issues.append(Issue(f"no module named '{patch.target}'", field=f"{where}.target"))
            continue
        if isinstance(patch, RemovePatch):
            set_submodule(model, patch.target, nn.Identity())
            continue

        try:
            in_shape, out_shape = capture_shapes(model, input_shape, [patch.target])[patch.target]
        except (KeyError, RuntimeError, TypeError, ValueError, AssertionError):
            issues.append(
                Issue(
                    "this model cannot be shape-checked on the meta device, so only freeze, "
                    "unfreeze and remove patches are available",
                    field=f"{where}.op",
                )
            )
            continue
        try:
            if isinstance(patch, ReplacePatch):
                chain, _ = build_chain(
                    patch.with_, in_shape, symbols, materialize, field_prefix=f"{where}.with"
                )
                set_submodule(model, patch.target, chain)
            elif isinstance(patch, InsertAfterPatch):
                chain, _ = build_chain(
                    patch.ops, out_shape, symbols, materialize, field_prefix=f"{where}.ops"
                )
                original = model.get_submodule(patch.target)
                set_submodule(model, patch.target, nn.Sequential(original, *chain))
        except SpecError as error:
            issues.extend(error.issues)
    if issues:
        raise SpecError(issues)
