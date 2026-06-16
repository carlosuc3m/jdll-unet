"""JSON-serializable validation metrics."""

from __future__ import annotations

import numpy as np
import torch

from .losses import Logits, primary_logits

try:  # pragma: no cover
    from scipy import ndimage as ndi
except Exception:  # pragma: no cover
    ndi = None


def _safe_float(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def binary_metrics(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> dict[str, float]:
    probs = torch.sigmoid(logits)
    pred = probs >= threshold
    target_bool = target.bool()
    intersection = (pred & target_bool).sum().float()
    union = (pred | target_bool).sum().float()
    denom = pred.sum().float() + target_bool.sum().float()
    dice = (2 * intersection + 1e-6) / (denom + 1e-6)
    iou = (intersection + 1e-6) / (union + 1e-6)
    return {"dice": _safe_float(dice), "iou": _safe_float(iou)}


def multiclass_metrics(logits: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred = torch.argmax(logits, dim=1)
    classes = int(logits.shape[1])
    result: dict[str, float] = {}
    dices: list[float] = []
    for cls in range(1, classes):
        pred_cls = pred == cls
        target_cls = target == cls
        intersection = (pred_cls & target_cls).sum().float()
        denom = pred_cls.sum().float() + target_cls.sum().float()
        dice = _safe_float((2 * intersection + 1e-6) / (denom + 1e-6))
        result[f"dice_class_{cls}"] = dice
        dices.append(dice)
    result["mean_dice"] = float(np.mean(dices)) if dices else 1.0
    return result


def instance_metrics(logits: torch.Tensor, target: dict[str, torch.Tensor]) -> dict[str, float]:
    metrics = binary_metrics(logits[:, 0:1], target["foreground"])
    metrics = {f"foreground_{key}": value for key, value in metrics.items()}
    boundary_loss_proxy = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[:, 1:2],
        target["boundary"].float(),
    )
    metrics["boundary_loss"] = _safe_float(boundary_loss_proxy)
    if ndi is not None:
        pred = (torch.sigmoid(logits[:, 0:1]) >= 0.5).detach().cpu().numpy()
        counts = []
        for item in pred[:, 0]:
            _, count = ndi.label(item)
            counts.append(count)
        metrics["object_count_estimate"] = float(np.mean(counts)) if counts else 0.0
    return metrics


def compute_metrics(
    task: str,
    logits: Logits,
    target: torch.Tensor | dict[str, torch.Tensor],
) -> dict[str, float]:
    logits = primary_logits(logits)
    if task == "binary_semantic":
        assert isinstance(target, torch.Tensor)
        return binary_metrics(logits, target)
    if task == "multiclass_semantic":
        assert isinstance(target, torch.Tensor)
        return multiclass_metrics(logits, target)
    if task == "instance_friendly":
        assert isinstance(target, dict)
        return instance_metrics(logits, target)
    raise ValueError(f"Unsupported task: {task}")


def primary_metric(task: str, metrics: dict[str, float]) -> float:
    if task == "binary_semantic":
        return metrics.get("dice", 0.0)
    if task == "multiclass_semantic":
        return metrics.get("mean_dice", 0.0)
    if task == "instance_friendly":
        return metrics.get("foreground_dice", 0.0)
    return 0.0
