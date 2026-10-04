"""Streaming validation reductions with explicit voxel/sample aggregation units."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np
import torch

from .errors import TrainingError
from .losses import Logits, Target, compute_loss, primary_logits, resize_target_for_logits
from .metrics import compute_metrics


class ValidationAccumulator:
    """Accumulate sufficient statistics, never predictions or computation graphs."""

    def __init__(self, task: str, weights: dict[str, float] | None = None,
                 focal_gamma: float = 2.0, focal_alpha: float | None = None) -> None:
        self.task = task
        self.weights = weights or {}
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.loss_sums: list[dict[str, float]] = []
        self.loss_counts: list[dict[str, float]] = []
        self.soft_dice: list[np.ndarray | None] = []
        self.metric_sums: dict[str, float] = defaultdict(float)
        self.metric_counts: dict[str, float] = defaultdict(float)
        self.hard_counts: dict[str, np.ndarray] = {}
        self.samples = 0
        self.batches = 0

    @staticmethod
    def _support(target: Target, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        valid = target.get("valid") if isinstance(target, dict) else None
        if valid is None:
            valid = torch.ones_like(logits[:, :1], dtype=torch.bool)
        semantic = target.get("foreground", target.get("semantic")) if isinstance(target, dict) else target
        assert isinstance(semantic, torch.Tensor)
        if semantic.ndim == valid.ndim - 1:
            semantic = semantic[:, None]
        return valid.bool(), semantic

    def update(self, logits: Logits, target: Target, *, cpu_validity: torch.Tensor | None = None) -> None:
        heads = list(logits) if isinstance(logits, (tuple, list)) else [logits]
        if self.loss_sums and len(heads) != len(self.loss_sums):
            raise TrainingError("Validation output-head count changed during an epoch")
        for index, head in enumerate(heads):
            if not bool(torch.isfinite(head).all()):
                raise TrainingError("Non-finite validation prediction")
            current = target if index == 0 else resize_target_for_logits(self.task, target, head)
            support = cpu_validity if index == 0 else None
            total, values = compute_loss(self.task, head, current, self.weights,
                                        self.focal_gamma, self.focal_alpha, cpu_validity=support)
            if not bool(torch.isfinite(total)):
                raise TrainingError("Non-finite validation loss")
            valid, semantic = self._support(current, head)
            counts = torch.stack([valid.sum(), (valid & (semantic > 0)).sum()]).cpu().tolist()
            valid_count, foreground = map(float, counts)
            if len(self.loss_sums) <= index:
                self.loss_sums.append(defaultdict(float))
                self.loss_counts.append(defaultdict(float))
                self.soft_dice.append(None)
            numbers = torch.stack(list(values.values())).detach().cpu().double().tolist()
            for key, value in zip(values, numbers, strict=True):
                count = (head.shape[0] if "dice_loss" in key else foreground if key == "distance_loss"
                         else valid_count - foreground if key == "distance_background_loss" else valid_count)
                self.loss_sums[index][key] += float(value) * count
                self.loss_counts[index][key] += count
            if self.task == "multiclass_semantic":
                probabilities = head.float().softmax(1)
                sums = []
                for cls in range(1, head.shape[1]):
                    truth = (semantic[:, 0] == cls) & valid[:, 0]
                    prob = probabilities[:, cls] * valid[:, 0]
                    sums.append(torch.stack([(prob * truth).sum(dtype=torch.float32),
                                             prob.sum(dtype=torch.float32) + truth.sum()]))
                stats = torch.stack(sums).detach().cpu().double().numpy()
                self.soft_dice[index] = stats if self.soft_dice[index] is None else self.soft_dice[index] + stats
        main = primary_logits(logits).float()
        valid, semantic = self._support(target, main)
        valid_count = float(valid.sum().item())
        foreground = float((valid & (semantic > 0)).sum().item())
        metrics = compute_metrics(self.task, main, target)
        for key, metric_value in metrics.items():
            if not math.isfinite(metric_value):
                raise TrainingError(f"Non-finite validation metric: {key}")
            count = valid_count if key == "boundary_loss" else foreground if key == "distance_mae" else main.shape[0]
            self.metric_sums[key] += metric_value * count
            self.metric_counts[key] += count
        if self.task == "multiclass_semantic":
            prediction = main.argmax(1, keepdim=True)
            for cls in range(1, main.shape[1]):
                self._hard(f"dice_class_{cls}", prediction == cls, semantic == cls, valid)
        else:
            self._hard("foreground" if self.task == "instance_friendly" else "binary",
                       main[:, :1] >= 0, semantic > 0, valid)
            if self.task == "instance_friendly":
                assert isinstance(target, dict)
                self._hard("boundary", main[:, 1:2] >= 0, target["boundary"] > 0, valid)
        self.samples += main.shape[0]
        self.batches += 1

    def _hard(self, name: str, prediction: torch.Tensor, truth: torch.Tensor, valid: torch.Tensor) -> None:
        prediction, truth = prediction & valid, truth & valid
        values = torch.stack([(prediction & truth).sum(), prediction.sum() + truth.sum(),
                              (prediction | truth).sum()]).cpu().numpy().astype(np.float64)
        self.hard_counts[name] = self.hard_counts.get(name, np.zeros(3)) + values

    def _head_losses(self, index: int) -> tuple[float, dict[str, float]]:
        losses = {key: value / max(1, self.loss_counts[index][key]) for key, value in self.loss_sums[index].items()}
        soft = self.soft_dice[index]
        if soft is not None:
            losses["dice_loss"] = float(1 - np.mean((2 * soft[:, 0] + 1e-6) / (soft[:, 1] + 1e-6)))
        names = {
            "bce_loss": ("bce", 1.0), "foreground_bce_loss": ("bce", 1.0),
            "dice_loss": ("dice", 1.0), "foreground_dice_loss": ("dice", 1.0),
            "cross_entropy_loss": ("cross_entropy", 1.0), "boundary_loss": ("boundary", 0.5),
            "distance_loss": ("distance", 1.0), "distance_background_loss": ("distance_background", 0.05),
            "focal_loss": ("focal", 0.0), "foreground_focal_loss": ("focal", 0.0),
            "boundary_focal_loss": ("boundary_focal", 0.0),
        }
        total = sum(value * self.weights.get(names[key][0], names[key][1]) for key, value in losses.items())
        return total, losses

    def result(self) -> tuple[dict[str, float], dict[str, float]]:
        if not self.samples:
            raise TrainingError("Validation completed without any samples")
        main, losses = self._head_losses(0)
        weighted, weight_sum = main, 1.0
        auxiliary = 0.0
        for index in range(1, len(self.loss_sums)):
            weight = 0.5**index
            auxiliary += weight * self._head_losses(index)[0]
            weight_sum += weight
        weighted += auxiliary
        if len(self.loss_sums) > 1:
            losses["deep_supervision_loss"] = auxiliary / (weight_sum - 1)
        losses["total_loss"] = weighted / weight_sum
        metrics = {key: value / max(1, self.metric_counts[key]) for key, value in self.metric_sums.items()}
        for key, (intersection, denominator, union) in self.hard_counts.items():
            dice = float((2 * intersection + 1e-6) / (denominator + 1e-6))
            if key.startswith("dice_class_"):
                metrics[key] = dice
            elif key == "boundary":
                metrics["boundary_f1"] = dice
            else:
                prefix = "foreground_" if key == "foreground" else ""
                metrics[prefix + "dice"] = dice
                metrics[prefix + "iou"] = float((intersection + 1e-6) / (union + 1e-6))
        if self.task == "multiclass_semantic":
            metrics["mean_dice"] = float(np.mean([value for key, value in metrics.items() if key.startswith("dice_class_")]))
        if not all(math.isfinite(value) for value in [*losses.values(), *metrics.values()]):
            raise TrainingError("Non-finite aggregated validation result")
        return losses, metrics

    @staticmethod
    def aggregation() -> dict[str, Any]:
        return {"dice_iou": "pooled_valid_voxels; multiclass macro-average excludes background",
                "instance_metrics": "mean_per_patch", "binary_dice_loss": "mean_per_patch",
                "multiclass_dice_loss": "pooled_valid_voxels_per_class",
                "pointwise_losses": "valid_voxel_weighted; distance uses foreground/background support",
                "deep_supervision": "same reductions per head, then normalized geometric weights"}
