"""Predictions from a trained model: deployments, parity checks and the playground (PLAN §8.5).

A model directory holds `model.json` (what the model is and how its inputs are prepared) and
`best.pt` (its weights). The model is rebuilt without downloading anything: its structure is
built on the meta device and the trained weights are assigned into it.

Parity: after training, a few sample inputs are written to `parity/` together with the
predictions the training environment made for them (`parity.json`). A deployment must reproduce
those predictions in its own process before it serves traffic.
"""

from __future__ import annotations

import base64
import dataclasses
import io
import itertools
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pydantic import TypeAdapter
from torch.nn import functional

from netpattern_engine import __version__
from netpattern_engine.data import readers
from netpattern_engine.data.dataset import adapt_channels, load_image
from netpattern_engine.data.index import Sample
from netpattern_engine.data.transforms import ModelInput, build_transforms
from netpattern_engine.errors import EngineError
from netpattern_engine.models.build import build_model
from netpattern_engine.spec import ArchitectureSpec, DatasetSpec, TaskSpec
from netpattern_engine.spec.transforms import TransformsSpec
from netpattern_engine.tasks import get_task
from netpattern_engine.tasks.base import TaskPlugin, main_output
from netpattern_engine.training.checkpoint import load_checkpoint

MODEL_FILE = "model.json"
WEIGHTS_FILE = "best.pt"
PARITY_FILE = "parity.json"
PARITY_DIR = "parity"
PARITY_SAMPLES = 4
# Allowed difference per number (probabilities, values, box coordinates): CPU and GPU kernels
# differ slightly, real mistakes (other weights, other preprocessing) differ far more.
PARITY_TOLERANCE = 1e-3

_ARCHITECTURE: TypeAdapter[Any] = TypeAdapter(ArchitectureSpec)
_DATASET: TypeAdapter[Any] = TypeAdapter(DatasetSpec)


def describe_model(
    pipeline: Any, task: TaskPlugin, model_input: ModelInput, data_channels: int
) -> dict[str, Any]:
    """The contents of model.json: everything needed to rebuild and feed the model."""
    return {
        "format": "netpattern-model/1",
        "engine": __version__,
        "task": pipeline.task.model_dump(mode="json"),
        "task_state": task.state(),
        "dataset": pipeline.data.model_dump(mode="json"),
        "transforms": pipeline.transforms.model_dump(mode="json"),
        "architecture": pipeline.architecture.model_dump(mode="json", by_alias=True),
        "input": dataclasses.asdict(model_input),
        "data_channels": data_channels,
    }


def _restore_task(description: dict[str, Any]) -> TaskPlugin:
    task = get_task(TaskSpec.model_validate(description["task"]))
    state = description["task_state"]
    task.setup(state.get("classes", []), state.get("targets", []), [])
    if "target_mean" in state:  # regression normalizes its targets with training statistics
        task.mean = torch.tensor(state["target_mean"])
        task.std = torch.tensor(state["target_std"])
    return task


def _model_input(raw: dict[str, Any]) -> ModelInput:
    return ModelInput(
        int(raw["channels"]),
        tuple(raw["size"]),
        int(raw["resize"]),
        tuple(raw["mean"]),
        tuple(raw["std"]),
    )


def _png_base64(array: np.ndarray) -> str:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


class Predictor:
    def __init__(
        self,
        module: torch.nn.Module,
        task: TaskPlugin,
        model_input: ModelInput,
        transforms: TransformsSpec,
        data_spec: Any,
        device: torch.device | str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.module = module.to(self.device).eval()
        self.task = task
        self.model_input = model_input
        self.data_spec = data_spec
        self.transform = build_transforms(transforms, task.type, model_input, train=False)
        # Presets other than "none" normalize with the model's mean/std (not for detection).
        self.normalized = transforms.preset != "none" and task.type != "detection.bbox"

    @classmethod
    def load(cls, model_dir: Path, device: str = "cpu") -> Predictor:
        description = json.loads((model_dir / MODEL_FILE).read_text(encoding="utf-8"))
        task = _restore_task(description)
        arch = _ARCHITECTURE.validate_python(description["architecture"])
        built = build_model(arch, task, int(description["data_channels"]), materialize=False)
        weights = model_dir / WEIGHTS_FILE
        if not weights.exists():  # a run directory keeps it under checkpoints/
            weights = model_dir / "checkpoints" / WEIGHTS_FILE
        state = load_checkpoint(weights)["model_state"]
        built.module.load_state_dict(state, assign=True)
        missing = [
            name
            for name, tensor in itertools.chain(
                built.module.named_parameters(), built.module.named_buffers()
            )
            if tensor.is_meta
        ]
        if missing:
            raise EngineError(f"the checkpoint does not cover {', '.join(missing[:5])}")
        return cls(
            built.module,
            task,
            _model_input(description["input"]),
            TransformsSpec.model_validate(description["transforms"]),
            _DATASET.validate_python(description["dataset"]),
            device,
        )

    # ----------------------------------------------------------------------- prediction

    def _prepare(self, path: Path) -> tuple[torch.Tensor, tuple[int, int]]:
        if readers.file_kind(path) == "nifti":
            raise EngineError("send 2D images (PNG, JPEG, TIFF, DICOM), not NIfTI volumes")
        from torchvision import tv_tensors

        array = adapt_channels(
            load_image(Sample(path=str(path)), self.data_spec, self.task.type),
            self.model_input.channels,
        )
        image = tv_tensors.Image(torch.from_numpy(np.ascontiguousarray(array)))
        return self.transform(image).as_subclass(torch.Tensor), tuple(array.shape[-2:])

    def predict(self, paths: list[Path], options: dict[str, Any] | None = None) -> dict[str, Any]:
        options = options or {}
        started = time.perf_counter()
        outputs = []
        # One image at a time: images may differ in size (detection, segmentation).
        for path in paths:
            image, size = self._prepare(Path(path))
            with torch.no_grad():
                if self.task.type == "detection.bbox":
                    raw = self.module([image.to(self.device)])
                else:
                    raw = self.module(image.unsqueeze(0).to(self.device))
            output = self._decode(raw, size, options)
            if options.get("explain"):
                output["explanation"] = self.explain(image, raw)
            outputs.append(output)
        return {
            "task": self.task.type,
            "outputs": outputs,
            "timing_ms": {"inference": round((time.perf_counter() - started) * 1000, 2)},
        }

    def explain(self, image: torch.Tensor, raw: Any) -> dict[str, Any]:
        """Grad-CAM on the last convolution: where the image pushed the top class up.

        Returns the heatmap and the image as the model saw it (after resizing), both as PNG.
        """
        if not self.task.type.startswith("classification"):
            return {"method": "grad-cam", "error": "explanations are available for classification"}
        layers = [(n, m) for n, m in self.module.named_modules() if isinstance(m, torch.nn.Conv2d)]
        if not layers:
            return {"method": "grad-cam", "error": "the model has no convolution layer"}
        name, layer = layers[-1]
        target = int(main_output(raw)[0].argmax())
        store: dict[str, torch.Tensor] = {}

        def keep(module: Any, inputs: Any, output: torch.Tensor) -> torch.Tensor:
            output = output.clone()  # later in-place layers must not overwrite what we read
            output.register_hook(lambda grad: store.__setitem__("grad", grad))
            store["activation"] = output
            return output

        handle = layer.register_forward_hook(keep)
        try:
            with torch.enable_grad():
                batch = image.unsqueeze(0).to(self.device).requires_grad_(True)
                score = main_output(self.module(batch))[0, target]
                score.backward()
        finally:
            handle.remove()
        activation, grad = store["activation"][0].detach(), store["grad"][0]
        cam = torch.relu((grad.mean(dim=(1, 2))[:, None, None] * activation).sum(dim=0))
        cam = cam / cam.max() if float(cam.max()) > 0 else cam
        cam = functional.interpolate(
            cam[None, None], size=image.shape[-2:], mode="bilinear", align_corners=False
        )[0, 0]
        seen = image.detach().cpu()
        if self.normalized:
            mean = torch.tensor(self.model_input.mean).view(-1, 1, 1)
            std = torch.tensor(self.model_input.std).view(-1, 1, 1)
            seen = seen * std + mean
        pixels = (seen.clamp(0, 1).numpy() * 255).round().astype(np.uint8)
        pixels = pixels[0] if pixels.shape[0] == 1 else pixels[:3].transpose(1, 2, 0)
        return {
            "method": "grad-cam",
            "layer": name,
            "class": self.task.classes[target],
            "heatmap_png_base64": _png_base64((cam.cpu().numpy() * 255).round().astype(np.uint8)),
            "input_png_base64": _png_base64(np.ascontiguousarray(pixels)),
        }

    def _decode(self, raw: Any, size: tuple[int, int], options: dict[str, Any]) -> dict[str, Any]:
        kind, classes = self.task.type, self.task.classes
        if kind == "classification.single_label":
            scores = torch.softmax(main_output(raw)[0].float(), dim=0).cpu()
            k = max(1, min(int(options.get("top_k", 3)), len(classes)))
            values, indices = scores.topk(k)
            ranked = [
                {"label": classes[i], "score": round(float(v), 6)}
                for v, i in zip(values.tolist(), indices.tolist(), strict=True)
            ]
            return {"label": ranked[0]["label"], "score": ranked[0]["score"], "top_k": ranked}
        if kind == "classification.multi_label":
            scores = torch.sigmoid(main_output(raw)[0].float()).cpu().tolist()
            threshold = float(options.get("threshold", self.task.spec.threshold))
            return {
                "labels": [
                    {"label": name, "score": round(score, 6)}
                    for name, score in zip(classes, scores, strict=True)
                    if score >= threshold
                ],
                "scores": {name: round(s, 6) for name, s in zip(classes, scores, strict=True)},
            }
        if kind == "regression":
            values = self.task.denormalize(main_output(raw).float())[0].cpu().tolist()
            return {
                "values": {
                    name: round(v, 6) for name, v in zip(self.task.targets, values, strict=True)
                }
            }
        if kind == "segmentation.semantic":
            logits = raw["out"] if isinstance(raw, dict) else raw
            logits = functional.interpolate(
                logits.float(), size=size, mode="bilinear", align_corners=False
            )
            mask = logits.argmax(dim=1)[0].to(torch.uint8).cpu().numpy()
            counts = np.bincount(mask.ravel(), minlength=len(classes))
            return {
                "classes": classes,
                "pixel_fractions": {
                    name: round(float(c) / mask.size, 6)
                    for name, c in zip(classes, counts, strict=False)
                },
                # Pixel value = class index; same size as the input image.
                "mask_png_base64": _png_base64(mask),
            }
        if kind == "detection.bbox":
            found = raw[0]
            threshold = float(options.get("threshold", 0.5))
            detections = [
                {
                    "label": classes[int(label) - 1],
                    "score": round(float(score), 6),
                    "box": [round(float(v), 2) for v in box],
                }
                for box, label, score in zip(
                    found["boxes"].cpu(), found["labels"].cpu(), found["scores"].cpu(), strict=True
                )
                if float(score) >= threshold
            ]
            return {"detections": detections}
        raise EngineError(f"prediction is not available for {kind}")


# --------------------------------------------------------------------------- parity


def _numbers(value: Any) -> list[float]:
    """All numbers in an output, in a stable order."""
    if isinstance(value, bool):
        return [float(value)]
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, dict):
        return [n for key in sorted(value) for n in _numbers(value[key])]
    if isinstance(value, list):
        return [n for item in value for n in _numbers(item)]
    return []


def _texts(value: Any) -> list[str]:
    """Labels must match exactly; encoded images (masks) are compared through their numbers."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [
            t for key in sorted(value) if not key.endswith("_base64") for t in _texts(value[key])
        ]
    if isinstance(value, list):
        return [t for item in value for t in _texts(item)]
    return []


def write_parity(
    predictor: Predictor, samples: list[Sample], data_spec: Any, out_dir: Path
) -> list[dict[str, Any]]:
    """Save sample inputs as files and record what this environment predicts for them."""
    folder = out_dir / PARITY_DIR
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    records = []
    for number, sample in enumerate(samples[:PARITY_SAMPLES]):
        source = Path(sample.path)
        if sample.index is None and source.is_file():
            target = folder / f"{number}{source.suffix.lower()}"
            shutil.copyfile(source, target)
        else:
            # Synthetic samples and volume slices: store the pixels the model saw as an 8-bit PNG.
            from PIL import Image

            array = load_image(sample, data_spec, predictor.task.type)
            pixels = (np.clip(array, 0, 1) * 255).round().astype(np.uint8)
            pixels = pixels[0] if pixels.shape[0] == 1 else pixels[:3].transpose(1, 2, 0)
            target = folder / f"{number}.png"
            Image.fromarray(pixels).save(target)
        output = predictor.predict([target])["outputs"][0]
        records.append({"file": f"{PARITY_DIR}/{target.name}", "output": output})
    (out_dir / PARITY_FILE).write_text(json.dumps(records, indent=2), encoding="utf-8")
    return records


def check_parity(predictor: Predictor, model_dir: Path) -> dict[str, Any]:
    """Predict the stored parity samples again and compare with the training environment."""
    path = model_dir / PARITY_FILE
    if not path.exists():
        return {"passed": None, "samples": 0, "message": "this model has no parity samples"}
    records = json.loads(path.read_text(encoding="utf-8"))
    worst, mismatches = 0.0, []
    for record in records:
        output = predictor.predict([model_dir / record["file"]])["outputs"][0]
        expected, got = _numbers(record["output"]), _numbers(output)
        same_texts = _texts(record["output"]) == _texts(output)
        if len(expected) != len(got) or not same_texts:
            mismatches.append(record["file"])
            continue
        difference = max((abs(a - b) for a, b in zip(expected, got, strict=True)), default=0.0)
        worst = max(worst, difference)
        if difference > PARITY_TOLERANCE * max(1.0, max(map(abs, expected), default=1.0)):
            mismatches.append(record["file"])
    return {
        "passed": not mismatches,
        "samples": len(records),
        "max_difference": round(worst, 8),
        "mismatches": mismatches,
    }
