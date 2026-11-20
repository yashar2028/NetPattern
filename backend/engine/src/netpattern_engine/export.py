"""Export a model version for use outside NetPattern (PLAN Phase 6).

- state_dict: the trained weights (torch.save of model.state_dict()), plus model.json
- torchscript: a traced module that runs without NetPattern (torch.jit.load)
- onnx: for other runtimes, when the environment's PyTorch can export it

A format that fails is reported in `notes` instead of failing the whole export.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Callable

import torch

from netpattern_engine.inference import MODEL_FILE, Predictor

FORMATS = ("state_dict", "torchscript", "onnx")
EXPORT_DIR = "exports"


def export_model(
    model_dir: Path, run_dir: Path, formats: list[str], emit: Callable[..., None]
) -> dict[str, Any]:
    predictor = Predictor.load(model_dir, device="cpu")
    module, model_input = predictor.module.eval(), predictor.model_input
    out = run_dir / EXPORT_DIR
    out.mkdir(parents=True, exist_ok=True)
    example = torch.zeros(1, model_input.channels, *model_input.size)
    files: dict[str, str] = {}
    notes: dict[str, str] = {}
    detection = predictor.task.type == "detection.bbox"

    shutil.copyfile(model_dir / MODEL_FILE, out / MODEL_FILE)
    files["model.json"] = f"{EXPORT_DIR}/{MODEL_FILE}"
    for name in formats:
        emit("phase", name=f"exporting {name}")
        try:
            if name == "state_dict":
                torch.save(module.state_dict(), out / "weights.pt")
                files[name] = f"{EXPORT_DIR}/weights.pt"
            elif name == "torchscript":
                if detection:
                    raise ValueError("detection models take a list of images; use state_dict")
                with torch.no_grad():
                    traced = torch.jit.trace(module, example, strict=False)
                traced.save(str(out / "model.ts"))
                files[name] = f"{EXPORT_DIR}/model.ts"
            elif name == "onnx":
                if detection:
                    raise ValueError("detection models are not exported to ONNX yet")
                torch.onnx.export(
                    module,
                    (example,),
                    str(out / "model.onnx"),
                    dynamo=False,
                    input_names=["image"],
                    dynamic_axes={"image": {0: "batch"}},
                    opset_version=17,
                )
                files[name] = f"{EXPORT_DIR}/model.onnx"
            else:
                raise ValueError(f"unknown format (choose from {', '.join(FORMATS)})")
        except Exception as error:  # noqa: BLE001 - report per format
            notes[name] = f"{type(error).__name__}: {str(error).splitlines()[0][:300]}"
    return {
        "status": "completed",
        "files": files,
        "notes": notes,
        "input": {
            "shape": [model_input.channels, *model_input.size],
            "mean": list(model_input.mean),
            "std": list(model_input.std),
            "normalized": predictor.normalized,
        },
        "task": predictor.task.state(),
    }
