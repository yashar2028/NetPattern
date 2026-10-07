"""Interactive session (PLAN §4.4): JSON-RPC 2.0, one message per line on stdin/stdout.

Runs for as long as the editor is open, so validation and shape inference answer
in milliseconds instead of paying the PyTorch import on every edit.
"""

from __future__ import annotations

import base64
import io
import json
import sys
from typing import Any, Callable, TextIO

from pydantic import TypeAdapter, ValidationError

from netpattern_engine import __version__
from netpattern_engine.errors import EngineError, SpecError, issues_from_validation_error
from netpattern_engine.spec import ArchitectureSpec, DatasetSpec, TaskSpec
from netpattern_engine.spec.transforms import TransformsSpec

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
SPEC_ERROR = 4001
ENGINE_ERROR = 4002

_ARCHITECTURE = TypeAdapter(ArchitectureSpec)
_DATASET = TypeAdapter(DatasetSpec)


class RpcError(Exception):
    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(message)
        self.code, self.message, self.data = code, message, data


def _parse(adapter_or_model: Any, value: Any, field: str) -> Any:
    try:
        if hasattr(adapter_or_model, "validate_python"):
            return adapter_or_model.validate_python(value)
        return adapter_or_model.model_validate(value)
    except ValidationError as error:
        issues = issues_from_validation_error(error, prefix=field)
        raise RpcError(
            SPEC_ERROR, "invalid spec", {"issues": [i.to_dict() for i in issues]}
        ) from error


def _task_for(params: dict[str, Any]):
    from netpattern_engine.tasks import get_task

    task_spec = _parse(TaskSpec, params.get("task"), "task")
    task = get_task(task_spec)
    classes = params.get("classes") or task_spec.classes
    targets = params.get("targets") or task_spec.targets
    count = int(params.get("num_outputs") or 0)
    if task.type == "regression":
        targets = targets or [f"target_{i}" for i in range(max(count, 1))]
    elif task.type == "detection.bbox":
        classes = classes or [f"class_{i}" for i in range(max(count - 1, 1))]
    else:
        classes = classes or [f"class_{i}" for i in range(max(count, 2))]
    task.setup(list(classes or []), list(targets or []), [])
    return task


# --------------------------------------------------------------------------- methods


def ping(params: dict[str, Any]) -> dict[str, Any]:
    return {"pong": True, "engine": __version__}


def info(params: dict[str, Any]) -> dict[str, Any]:
    from netpattern_engine.runtime import environment_info

    return environment_info()


def schema(params: dict[str, Any]) -> dict[str, Any]:
    from netpattern_engine.schema import schema_document

    return schema_document()


def palette(params: dict[str, Any]) -> dict[str, Any]:
    from netpattern_engine.data.transforms import transform_palette
    from netpattern_engine.graph.ops import palette as graph_palette

    return {"graph_ops": graph_palette(), "transform_ops": transform_palette()}


def validate_architecture(params: dict[str, Any]) -> dict[str, Any]:
    """Shapes, parameter counts, FLOPs and per-node issues, computed on the meta device."""
    import torch

    from netpattern_engine.graph.compiler import analyze_graph, count_flops
    from netpattern_engine.models.build import build_model, symbols_for
    from netpattern_engine.spec.architecture import GraphArchitecture

    arch = _parse(_ARCHITECTURE, params.get("architecture"), "architecture")
    task = _task_for(params)
    channels = int(params.get("data_channels", 3))

    if isinstance(arch, GraphArchitecture):
        result = analyze_graph(arch, symbols_for(task), materialize=False)
        issues = [issue.to_dict() for issue in result.issues]
        if not issues and result.output_shape is not None:
            problem = task.check_output_shape(result.output_shape)
            if problem:
                issues.append({"message": problem, "node_id": result.output})
        return {
            "ok": not issues,
            "issues": issues,
            "input_shape": list(arch.input.shape),
            "output_shape": list(result.output_shape) if result.output_shape else None,
            "nodes": {
                node_id: {"shape": list(shape), "params": result.params.get(node_id, 0)}
                for node_id, shape in result.shapes.items()
            },
            "params": result.total_params,
            "trainable_params": result.total_params,
            "flops": result.flops,
        }

    try:
        built = build_model(arch, task, channels, materialize=False)
    except SpecError as error:
        return {"ok": False, "issues": [issue.to_dict() for issue in error.issues]}
    input_shape = (built.input.channels, *built.input.size)
    output_shape, flops, issues = None, None, []
    if task.type != "detection.bbox":
        try:
            built.module.eval()
            with torch.no_grad():
                out = built.module(torch.empty((2, *input_shape), device="meta"))
            from netpattern_engine.tasks.base import main_output

            output_shape = list(main_output(out).shape[1:])
            if task.type != "segmentation.semantic":
                problem = task.check_output_shape(tuple(output_shape))
                if problem:
                    issues.append({"message": problem})
        except Exception as error:  # noqa: BLE001 - report any shape failure to the editor
            issues.append({"message": f"forward pass failed: {str(error).splitlines()[0][:200]}"})
        flops = count_flops(built.module, input_shape)
    return {
        "ok": not issues,
        "issues": issues,
        "input_shape": list(input_shape),
        "output_shape": output_shape,
        "params": built.total_params,
        "trainable_params": built.trainable_params,
        "flops": flops,
        "head": built.head,
        "input": {
            "size": list(built.input.size),
            "resize": built.input.resize,
            "mean": list(built.input.mean),
            "std": list(built.input.std),
        },
    }


def inspect_model(params: dict[str, Any]) -> dict[str, Any]:
    from netpattern_engine.graph.inspect import inspect_model as inspect
    from netpattern_engine.models.build import build_model

    arch = _parse(_ARCHITECTURE, params.get("architecture"), "architecture")
    task = _task_for(params)
    built = build_model(arch, task, int(params.get("data_channels", 3)), materialize=False)
    return inspect(built.module, (built.input.channels, *built.input.size))


def list_zoo(params: dict[str, Any]) -> list[dict[str, Any]]:
    from netpattern_engine.models.zoo import list_models

    return list_models(
        task=params.get("task"),
        source=params.get("source"),
        query=params.get("query"),
        limit=int(params.get("limit", 50)),
    )


def model_card(params: dict[str, Any]) -> dict[str, Any]:
    from netpattern_engine.models.zoo import model_card as card

    try:
        return card(str(params.get("source")), str(params.get("name")))
    except KeyError as error:
        raise RpcError(INVALID_PARAMS, str(error)) from error


def inspect_dataset(params: dict[str, Any]) -> dict[str, Any]:
    from netpattern_engine.data.index import build_index
    from netpattern_engine.data.splits import split_samples

    data = _parse(_DATASET, params.get("dataset"), "dataset")
    task_spec = _parse(TaskSpec, params.get("task"), "task")
    index = build_index(data, task_spec)
    splits = split_samples(index.samples, data.split, task_spec.type)
    return {
        "samples": len(index.samples),
        "groups": len({s.group for s in index.samples}),
        "classes": index.classes,
        "targets": index.targets,
        "splits": {name: len(members) for name, members in splits.items()},
        "warnings": index.warnings,
    }


def preview_transforms(params: dict[str, Any]) -> dict[str, Any]:
    """The same sample under N random training augmentations, as PNG data URLs."""
    import numpy as np
    import torch
    from PIL import Image

    from netpattern_engine.data.dataset import ImageDataset
    from netpattern_engine.data.index import build_index
    from netpattern_engine.data.transforms import build_transforms
    from netpattern_engine.models.build import build_model

    data = _parse(_DATASET, params.get("dataset"), "dataset")
    task_spec = _parse(TaskSpec, params.get("task"), "task")
    transforms = _parse(TransformsSpec, params.get("transforms", {}), "transforms")
    arch = _parse(_ARCHITECTURE, params.get("architecture"), "architecture")
    count = max(1, min(int(params.get("count", 4)), 12))

    index = build_index(data, task_spec)
    task = _task_for(
        {"task": params.get("task"), "classes": index.classes, "targets": index.targets}
    )
    position = int(params.get("sample", 0)) % len(index.samples)
    built = build_model(arch, task, 3, materialize=False)
    transform = build_transforms(transforms, task.type, built.input, train=True)
    dataset = ImageDataset(
        [index.samples[position]],
        data,
        task.type,
        built.input.channels,
        task.num_outputs,
        transform,
    )
    mean = torch.tensor(built.input.mean).view(-1, 1, 1)
    std = torch.tensor(built.input.std).view(-1, 1, 1)
    torch.manual_seed(int(params.get("seed", 0)))
    images = []
    for _ in range(count):
        image, _ = dataset[0]
        if transforms.preset != "none" and task.type != "detection.bbox":
            image = image * std + mean
        array = (image.clamp(0, 1).numpy() * 255).astype(np.uint8)
        array = array[0] if array.shape[0] != 3 else array.transpose(1, 2, 0)
        buffer = io.BytesIO()
        Image.fromarray(array).save(buffer, format="PNG")
        images.append("data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode())
    return {"sample": index.samples[position].path, "images": images}


METHODS: dict[str, Callable[[dict[str, Any]], Any]] = {
    "ping": ping,
    "info": info,
    "schema": schema,
    "palette": palette,
    "validate_architecture": validate_architecture,
    "inspect_model": inspect_model,
    "list_zoo": list_zoo,
    "model_card": model_card,
    "inspect_dataset": inspect_dataset,
    "preview_transforms": preview_transforms,
}


# --------------------------------------------------------------------------- protocol


def handle(message: dict[str, Any]) -> dict[str, Any] | None:
    request_id = message.get("id")
    try:
        if message.get("jsonrpc") != "2.0" or not isinstance(message.get("method"), str):
            raise RpcError(INVALID_REQUEST, "expected a JSON-RPC 2.0 request")
        method = METHODS.get(message["method"])
        if method is None:
            raise RpcError(METHOD_NOT_FOUND, f"unknown method '{message['method']}'")
        params = message.get("params") or {}
        if not isinstance(params, dict):
            raise RpcError(INVALID_PARAMS, "params must be an object")
        result = method(params)
        return {"jsonrpc": "2.0", "id": request_id, "result": result}
    except RpcError as error:
        payload = {"code": error.code, "message": error.message}
        if error.data is not None:
            payload["data"] = error.data
        return {"jsonrpc": "2.0", "id": request_id, "error": payload}
    except SpecError as error:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": SPEC_ERROR, "message": str(error), "data": error.to_dict()},
        }
    except EngineError as error:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": ENGINE_ERROR, "message": str(error)},
        }
    except Exception as error:  # noqa: BLE001 - the session must survive any request
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": INTERNAL_ERROR, "message": f"{type(error).__name__}: {error}"[:500]},
        }


def serve(stdin: TextIO, stdout: TextIO) -> int:
    def reply(payload: dict[str, Any]) -> None:
        stdout.write(json.dumps(payload, default=str, separators=(",", ":")) + "\n")
        stdout.flush()

    reply({"jsonrpc": "2.0", "method": "ready", "params": {"engine": __version__}})
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError as error:
            reply(
                {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": PARSE_ERROR, "message": str(error)},
                }
            )
            continue
        if isinstance(message, dict) and message.get("method") == "shutdown":
            reply({"jsonrpc": "2.0", "id": message.get("id"), "result": {"bye": True}})
            return 0
        reply(handle(message if isinstance(message, dict) else {}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(serve(sys.stdin, sys.stdout))
