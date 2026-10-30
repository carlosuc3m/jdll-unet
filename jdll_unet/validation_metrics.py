"""Streaming validation reductions with explicit voxel/sample aggregation units."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

import numpy as np
import torch

from .config import PostprocessingConfig
from .errors import TrainingError
from .losses import Logits, Target, compute_loss, primary_logits, resize_target_for_logits
from .metrics import compute_metrics, instance_label_counts, instance_pq_statistics
from .postprocess import postprocess_instance


def reconstruction_candidates(config: PostprocessingConfig, count: int = 3) -> list[dict[str, Any]]:
    base = asdict(config)
    options = [base, {**base, "threshold": 0.5, "seed_distance_threshold": 0.15, "seed_h": 0.05},
               {**base, "threshold": 0.4, "seed_distance_threshold": 0.15, "seed_h": 0.05}]
    if config.method == "connected_components":
        options = [{**base, "threshold": option["threshold"]} for option in options]
    selected: list[dict[str, Any]] = []
    for option in options[:count]:
        if option not in selected:
            selected.append(option)
    return selected


class InstancePQAccumulator:
    """One shared reconstruction choice; counts only, never retained predictions."""

    def __init__(self, config: PostprocessingConfig, count: int = 3,
                 target_cache: dict | None = None, check_cancel: Callable[[], None] | None = None) -> None:
        self.configs = reconstruction_candidates(config, count)
        self.cases: dict[str, np.ndarray] = {}
        self.target_cache = target_cache if target_cache is not None else {}
        self.check_cancel = check_cancel

    def update_batch(self, logits: Logits, instances: torch.Tensor, cases: list[str], keys: list[Any],
                     valid: torch.Tensor | None = None, spacings: list[tuple[float, ...]] | None = None) -> None:
        probabilities = torch.sigmoid(primary_logits(logits).float()).detach().cpu().numpy()
        truths = instances.detach().cpu().numpy()
        support = valid.detach().cpu().numpy() if valid is not None else None
        for index in range(len(truths)):
            self.update(probabilities[index], truths[index, 0], cases[index], keys[index],
                        support[index, 0] if support is not None else None,
                        spacings[index] if spacings is not None else None)

    def update(self, probabilities: np.ndarray, target: np.ndarray, case: str, key: Any,
               valid: np.ndarray | None = None, spacing: tuple[float, ...] | None = None) -> None:
        if valid is not None and not np.all(valid):
            probabilities = probabilities.copy()
            probabilities[:, ~valid] = 0
            probabilities[0, ~valid] = -1
            target = np.where(valid, target, 0)
        cache_key = (key, target.shape)
        counts = self.target_cache.get(cache_key)
        if counts is None:
            counts = instance_label_counts(target)
            # Bound cached metadata even for unusually crowded datasets.
            if len(counts) <= 100_000:
                while self.target_cache and (len(self.target_cache) >= 1024
                      or sum(len(value) for value in self.target_cache.values()) + len(counts) > 100_000):
                    self.target_cache.pop(next(iter(self.target_cache)))
                self.target_cache[cache_key] = counts
        statistics = self.cases.setdefault(case, np.zeros((len(self.configs), 3), dtype=np.float64))
        for index, config in enumerate(self.configs):
            if self.check_cancel is not None:
                self.check_cancel()
            prediction = postprocess_instance(probabilities[0], probabilities[1], probabilities[2],
                                               **{**config, "spacing": spacing})["labels"]
            if valid is not None:
                prediction[~valid] = 0
            statistics[index] += instance_pq_statistics(prediction, target, counts)
            del prediction

    def result(self) -> dict[str, Any]:
        if not self.cases:
            raise TrainingError("Instance validation completed without samples")
        scores: list[dict[str, Any]] = []
        for index, config in enumerate(self.configs):
            per_case = {}
            for case, statistics in self.cases.items():
                overlap, truths, predictions = statistics[index]
                denominator = 0.5 * (truths + predictions)
                # Empty patches add zero statistics; only a whole empty source
                # with no false positives receives the perfect-empty score.
                per_case[case] = float(overlap / denominator) if denominator else 1.0
            scores.append({"postprocessing": config, "panoptic_quality": float(np.mean(list(per_case.values()))),
                           "per_case": per_case})
        best = max(range(len(scores)), key=lambda index: scores[index]["panoptic_quality"])
        return {**scores[best], "candidate_index": best, "candidates": scores,
                "matching_iou_threshold": 0.5, "aggregation": "pooled_matching_within_source_then_mean_per_source"}


class ValidationAccumulator:
    """Accumulate sufficient statistics, never predictions or computation graphs."""

    def __init__(self, task: str, weights: dict[str, float] | None = None,
                 focal_gamma: float = 2.0, focal_alpha: float | None = None,
                 instance_validation: InstancePQAccumulator | None = None) -> None:
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
        self.instance_validation = instance_validation

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

    def update(self, logits: Logits, target: Target, *, cpu_validity: torch.Tensor | None = None,
               cpu_instances: torch.Tensor | None = None, cases: list[str] | None = None,
               keys: list[Any] | None = None, spacings: list[tuple[float, ...]] | None = None) -> None:
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
        if self.instance_validation is not None:
            assert isinstance(target, dict)
            truths = cpu_instances if cpu_instances is not None else target["instances"]
            support = cpu_validity if cpu_validity is not None else target.get("valid")
            samples = list(range(self.samples, self.samples + main.shape[0]))
            self.instance_validation.update_batch(main, truths, cases if cases is not None else [str(i) for i in samples],
                                                  keys if keys is not None else samples, support, spacings)
            self.samples += main.shape[0]
            self.batches += 1
            return
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
        if self.instance_validation is not None:
            metrics = {"panoptic_quality": self.instance_validation.result()["panoptic_quality"]}
            if not all(math.isfinite(value) for value in [*losses.values(), *metrics.values()]):
                raise TrainingError("Non-finite aggregated validation result")
            return losses, metrics
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

    def aggregation(self) -> dict[str, Any]:
        return {"dice_iou": "pooled_valid_voxels; multiclass macro-average excludes background",
                "instance_metrics": "pooled_matching_within_source_then_mean_per_source"
                if self.instance_validation is not None else "mean_per_patch", "binary_dice_loss": "mean_per_patch",
                "multiclass_dice_loss": "pooled_valid_voxels_per_class",
                "pointwise_losses": "valid_voxel_weighted; distance uses foreground/background support",
                "deep_supervision": "same reductions per head, then normalized geometric weights"}
