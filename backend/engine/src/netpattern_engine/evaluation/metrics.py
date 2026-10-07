"""Tier-1 metrics per task (PLAN §7.7).

Each metric object accumulates batches with `update(preds, targets)` and returns
`(scalars, details)` from `compute()`: scalars are plain floats for curves and
leaderboards, details hold tables such as the confusion matrix or per-class scores.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from netpattern_engine.data.readers import require_module


def _safe_div(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    return torch.where(
        denominator > 0, numerator / denominator.clamp(min=1e-12), torch.zeros_like(numerator)
    )


def _round(value: float) -> float | None:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return None
    return round(float(value), 6)


class ConfusionMetrics:
    """Single-label classification and semantic segmentation share a confusion matrix."""

    def __init__(self, classes: list[str], ignore_index: int | None = None) -> None:
        self.classes = classes
        self.ignore_index = ignore_index
        n = len(classes)
        self.matrix = torch.zeros((n, n), dtype=torch.int64)

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        preds, targets = preds.reshape(-1).cpu(), targets.reshape(-1).cpu()
        n = len(self.classes)
        keep = (targets >= 0) & (targets < n)
        if self.ignore_index is not None:
            keep &= targets != self.ignore_index
        flat = targets[keep] * n + preds[keep].clamp(0, n - 1)
        self.matrix += torch.bincount(flat, minlength=n * n).reshape(n, n)

    def _per_class(self) -> dict[str, torch.Tensor]:
        matrix = self.matrix.double()
        true_positive = matrix.diag()
        support = matrix.sum(dim=1)
        predicted = matrix.sum(dim=0)
        precision = _safe_div(true_positive, predicted)
        recall = _safe_div(true_positive, support)
        f1 = _safe_div(2 * true_positive, support + predicted)
        iou = _safe_div(true_positive, support + predicted - true_positive)
        return {
            "tp": true_positive,
            "support": support,
            "predicted": predicted,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": iou,
        }


class SingleLabelMetrics(ConfusionMetrics):
    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        super().update(preds.argmax(dim=1), targets)

    def compute(self) -> tuple[dict[str, Any], dict[str, Any]]:
        stats = self._per_class()
        total = stats["support"].sum()
        present = stats["support"] > 0
        scalars = {
            "accuracy": _round((stats["tp"].sum() / total).item()) if total else None,
            "balanced_accuracy": (
                _round(stats["recall"][present].mean().item()) if present.any() else None
            ),
            "precision_macro": (
                _round(stats["precision"][present].mean().item()) if present.any() else None
            ),
            "recall_macro": (
                _round(stats["recall"][present].mean().item()) if present.any() else None
            ),
            "f1_macro": _round(stats["f1"][present].mean().item()) if present.any() else None,
        }
        details = {
            "confusion_matrix": {"labels": self.classes, "matrix": self.matrix.tolist()},
            "per_class": [
                {
                    "class": name,
                    "precision": _round(stats["precision"][i].item()),
                    "recall": _round(stats["recall"][i].item()),
                    "f1": _round(stats["f1"][i].item()),
                    "support": int(stats["support"][i].item()),
                }
                for i, name in enumerate(self.classes)
            ],
        }
        return scalars, details


class SegmentationMetrics(ConfusionMetrics):
    def compute(self) -> tuple[dict[str, Any], dict[str, Any]]:
        stats = self._per_class()
        total = stats["support"].sum()
        union = stats["support"] + stats["predicted"] - stats["tp"]
        present = union > 0
        foreground = present.clone()
        if len(foreground) > 1:
            foreground[0] = False
        scalars = {
            "miou": _round(stats["iou"][present].mean().item()) if present.any() else None,
            "dice": _round(stats["f1"][present].mean().item()) if present.any() else None,
            "foreground_dice": (
                _round(stats["f1"][foreground].mean().item()) if foreground.any() else None
            ),
            "pixel_accuracy": _round((stats["tp"].sum() / total).item()) if total else None,
        }
        details = {
            "per_class": [
                {
                    "class": name,
                    "iou": _round(stats["iou"][i].item()),
                    "dice": _round(stats["f1"][i].item()),
                    "pixels": int(stats["support"][i].item()),
                }
                for i, name in enumerate(self.classes)
            ]
        }
        return scalars, details


class MultiLabelMetrics:
    def __init__(self, classes: list[str], threshold: float) -> None:
        self.classes = classes
        self.threshold = threshold
        self.probabilities: list[torch.Tensor] = []
        self.targets: list[torch.Tensor] = []

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        self.probabilities.append(torch.sigmoid(preds.detach().float()).cpu())
        self.targets.append(targets.detach().cpu())

    def compute(self) -> tuple[dict[str, Any], dict[str, Any]]:
        from torchmetrics.functional.classification import multilabel_average_precision

        probabilities = torch.cat(self.probabilities)
        targets = torch.cat(self.targets).long()
        predicted = (probabilities >= self.threshold).long()
        tp = (predicted * targets).sum(dim=0).double()
        fp = (predicted * (1 - targets)).sum(dim=0).double()
        fn = ((1 - predicted) * targets).sum(dim=0).double()
        f1 = _safe_div(2 * tp, 2 * tp + fp + fn)
        support = targets.sum(dim=0)
        present = support > 0
        micro = _safe_div(2 * tp.sum(), 2 * tp.sum() + fp.sum() + fn.sum())
        try:
            mean_ap = multilabel_average_precision(
                probabilities, targets, num_labels=len(self.classes), average="macro"
            ).item()
        except Exception:
            mean_ap = float("nan")
        scalars = {
            "f1_macro": _round(f1[present].mean().item()) if present.any() else None,
            "f1_micro": _round(micro.item()),
            "map": _round(mean_ap),
            "hamming_loss": _round((predicted != targets).double().mean().item()),
            "exact_match": _round((predicted == targets).all(dim=1).double().mean().item()),
        }
        details = {
            "per_class": [
                {"class": name, "f1": _round(f1[i].item()), "support": int(support[i].item())}
                for i, name in enumerate(self.classes)
            ]
        }
        return scalars, details


class RegressionMetrics:
    def __init__(self, targets: list[str]) -> None:
        self.names = targets
        self.preds: list[torch.Tensor] = []
        self.targets: list[torch.Tensor] = []

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        self.preds.append(preds.detach().double().cpu().reshape(len(preds), -1))
        self.targets.append(targets.detach().double().cpu().reshape(len(targets), -1))

    def compute(self) -> tuple[dict[str, Any], dict[str, Any]]:
        preds, targets = torch.cat(self.preds), torch.cat(self.targets)
        errors = preds - targets
        mae = errors.abs().mean(dim=0)
        mse = (errors**2).mean(dim=0)
        total = ((targets - targets.mean(dim=0)) ** 2).sum(dim=0)
        r2 = torch.where(
            total > 0, 1 - (errors**2).sum(dim=0) / total, torch.full_like(total, float("nan"))
        )
        scalars = {
            "mae": _round(mae.mean().item()),
            "mse": _round(mse.mean().item()),
            "rmse": _round(mse.sqrt().mean().item()),
            "r2": _round(r2.nanmean().item()) if not torch.isnan(r2).all() else None,
        }
        details = {
            "per_target": [
                {
                    "target": name,
                    "mae": _round(mae[i].item()),
                    "rmse": _round(mse[i].sqrt().item()),
                    "r2": _round(r2[i].item()),
                }
                for i, name in enumerate(self.names)
            ]
        }
        return scalars, details


class DetectionMetrics:
    def __init__(self, classes: list[str]) -> None:
        require_module("pycocotools", "detection", "Detection mAP")
        from torchmetrics.detection import MeanAveragePrecision

        self.classes = classes
        self.metric = MeanAveragePrecision(
            box_format="xyxy", iou_type="bbox", class_metrics=True, backend="pycocotools"
        )

    def update(
        self, preds: list[dict[str, torch.Tensor]], targets: list[dict[str, torch.Tensor]]
    ) -> None:
        self.metric.update(
            [{key: value.detach().cpu() for key, value in p.items()} for p in preds],
            [{key: value.detach().cpu() for key, value in t.items()} for t in targets],
        )

    def compute(self) -> tuple[dict[str, Any], dict[str, Any]]:
        result = self.metric.compute()
        scalars = {
            "map": _round(result["map"].item()),
            "map_50": _round(result["map_50"].item()),
            "map_75": _round(result["map_75"].item()),
            "mar_100": _round(result["mar_100"].item()),
        }
        per_class = []
        class_ids = result.get("classes")
        per_class_map = result.get("map_per_class")
        if class_ids is not None and per_class_map is not None and per_class_map.numel() > 1:
            for class_id, value in zip(
                class_ids.reshape(-1).tolist(), per_class_map.reshape(-1).tolist()
            ):
                if 1 <= class_id <= len(self.classes):
                    per_class.append({"class": self.classes[class_id - 1], "map": _round(value)})
        return scalars, {"per_class": per_class}
