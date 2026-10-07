"""torch Dataset over indexed samples: reads pixels, builds targets, applies transforms."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torchvision import tv_tensors

from netpattern_engine.data import readers, synthetic
from netpattern_engine.data.index import Sample
from netpattern_engine.errors import EngineError

LUMINANCE = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def load_image(sample: Sample, data_spec: Any, task_type: str) -> np.ndarray:
    """(C, H, W) float32 in [0, 1]."""
    if sample.path.startswith("synthetic:"):
        image, _, _ = synthetic.render(task_type, data_spec, int(sample.path.split(":", 1)[1]))
        return image
    if sample.index is not None:
        return readers.read_volume_slices(
            sample.path,
            data_spec.slices.axis,
            sample.index,
            data_spec.slices.context,
            data_spec.intensity,
        )
    kind = readers.file_kind(sample.path)
    if kind == "dicom":
        return readers.read_dicom_frame(sample.path, None, data_spec.intensity)
    if kind == "natural":
        return readers.read_natural(sample.path, data_spec.intensity)
    raise EngineError(f"unsupported image file: {sample.path}")


def load_mask(sample: Sample, data_spec: Any) -> np.ndarray:
    if sample.path.startswith("synthetic:"):
        _, _, mask = synthetic.render(
            "segmentation.semantic", data_spec, int(sample.path.split(":", 1)[1])
        )
        return mask
    if sample.mask is None:
        raise EngineError(f"no mask for {sample.path}")
    if readers.file_kind(sample.mask) == "nifti":
        mask = readers.read_mask_volume_slice(sample.mask, data_spec.slices.axis, sample.index or 0)
    else:
        mask = readers.read_mask_natural(sample.mask)
    return readers.remap_mask(mask, getattr(data_spec, "mask_mapping", None))


def adapt_channels(image: np.ndarray, channels: int) -> np.ndarray:
    have = image.shape[0]
    if have == channels:
        return image
    if have == 1:
        return np.repeat(image, channels, axis=0)
    if have == 3 and channels == 1:
        return np.tensordot(LUMINANCE, image, axes=1)[None].astype(np.float32)
    raise EngineError(
        f"images have {have} channels but the model expects {channels}; "
        "for 2.5D use slices.context=1 (3 channels) or a model with matching input channels"
    )


class ImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        samples: list[Sample],
        data_spec: Any,
        task_type: str,
        channels: int,
        num_outputs: int,
        transform: Any,
    ) -> None:
        self.samples = samples
        self.data_spec = data_spec
        self.task_type = task_type
        self.channels = channels
        self.num_outputs = num_outputs
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, Any]:
        sample = self.samples[i]
        array = adapt_channels(load_image(sample, self.data_spec, self.task_type), self.channels)
        image = tv_tensors.Image(torch.from_numpy(np.ascontiguousarray(array)))
        height, width = array.shape[-2:]

        if self.task_type == "classification.single_label":
            image, target = self.transform(image, torch.tensor(sample.label, dtype=torch.long))
        elif self.task_type == "classification.multi_label":
            multi_hot = torch.zeros(self.num_outputs, dtype=torch.float32)
            multi_hot[list(sample.label)] = 1.0
            image, target = self.transform(image, multi_hot)
        elif self.task_type == "regression":
            image, target = self.transform(image, torch.tensor(sample.label, dtype=torch.float32))
        elif self.task_type == "segmentation.semantic":
            mask = load_mask(sample, self.data_spec)
            if mask.shape != (height, width):
                raise EngineError(
                    f"mask size {mask.shape} differs from image size {(height, width)} "
                    f"for {sample.path}"
                )
            image, mask = self.transform(image, tv_tensors.Mask(torch.from_numpy(mask)))
            target = mask.as_subclass(torch.Tensor).long()
        elif self.task_type == "detection.bbox":
            boxes = torch.tensor(sample.boxes, dtype=torch.float32).reshape(-1, 4)
            target = {
                "boxes": tv_tensors.BoundingBoxes(
                    boxes, format="XYXY", canvas_size=(height, width)
                ),
                "labels": torch.tensor(sample.box_labels, dtype=torch.int64),
            }
            image, target = self.transform(image, target)
            target = {
                "boxes": target["boxes"].as_subclass(torch.Tensor),
                "labels": target["labels"],
            }
        else:
            raise EngineError(f"unknown task type {self.task_type}")
        return image.as_subclass(torch.Tensor), target
