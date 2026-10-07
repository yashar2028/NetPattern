"""Build models without allocating their weights.

A global `torch.device("meta")` context breaks constructors that compute with
tensors while building (RegNet widths, timm drop-path schedules call `.tolist()`).
Instead, constructors run normally and every parameter/buffer is moved to the meta
device the moment it is registered, so only shapes are kept.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

from torch import nn


@contextlib.contextmanager
def meta_init() -> Iterator[None]:
    register_parameter = nn.Module.register_parameter
    register_buffer = nn.Module.register_buffer

    def parameter_on_meta(module: nn.Module, name: str, param) -> None:
        register_parameter(module, name, param)
        if param is not None:
            stored = module._parameters[name]
            moved = type(stored)(stored.to("meta"), requires_grad=stored.requires_grad)
            moved.__dict__.update(stored.__dict__)
            module._parameters[name] = moved

    def buffer_on_meta(module: nn.Module, name: str, buffer, persistent: bool = True) -> None:
        register_buffer(module, name, buffer, persistent=persistent)
        if buffer is not None:
            module._buffers[name] = module._buffers[name].to("meta")

    nn.Module.register_parameter = parameter_on_meta  # type: ignore[method-assign]
    nn.Module.register_buffer = buffer_on_meta  # type: ignore[method-assign]
    try:
        yield
    finally:
        nn.Module.register_parameter = register_parameter  # type: ignore[method-assign]
        nn.Module.register_buffer = register_buffer  # type: ignore[method-assign]
