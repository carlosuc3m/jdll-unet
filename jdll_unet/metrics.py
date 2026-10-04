"""JSON-serializable validation metrics."""

from __future__ import annotations

import numpy as np
import torch

from .losses import Logits, masked_mean, primary_logits, valid_support
from .postprocess import _remove_small, postprocess_binary, postprocess_instance

try:  # pragma: no cover
    from scipy import ndimage as ndi
except Exception:  # pragma: no cover
    ndi = None


def _safe_float(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def binary_metrics(
    logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, valid: torch.Tensor | None = None,
    *, prediction: torch.Tensor | None = None,
) -> dict[str, float]:
    pred = torch.sigmoid(logits) >= threshold if prediction is None else prediction
    target_bool = target.bool()
    if valid is not None:
        pred = pred & valid
        target_bool = target_bool & valid
    intersection = (pred & target_bool).sum().float()
    union = (pred | target_bool).sum().float()
    denom = pred.sum().float() + target_bool.sum().float()
    dice = (2 * intersection + 1e-6) / (denom + 1e-6)
    iou = (intersection + 1e-6) / (union + 1e-6)
    return {"dice": _safe_float(dice), "iou": _safe_float(iou)}


def multiclass_metrics(
    logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor | None = None,
    *, prediction: torch.Tensor | None = None,
) -> dict[str, float]:
    pred = torch.argmax(logits, dim=1) if prediction is None else prediction
    classes = int(logits.shape[1])
    result: dict[str, float] = {}
    dices: list[float] = []
    for cls in range(1, classes):
        pred_cls = pred == cls
        target_cls = target == cls
        if valid is not None:
            pred_cls &= valid[:, 0]
            target_cls &= valid[:, 0]
        intersection = (pred_cls & target_cls).sum().float()
        denom = pred_cls.sum().float() + target_cls.sum().float()
        dice = _safe_float((2 * intersection + 1e-6) / (denom + 1e-6))
        result[f"dice_class_{cls}"] = dice
        dices.append(dice)
    result["mean_dice"] = float(np.mean(dices)) if dices else 1.0
    return result


def instance_metrics(logits: torch.Tensor, target: dict[str, torch.Tensor],
                     postprocessing: dict | None = None) -> dict[str, float]:
    valid = valid_support(target, logits)
    metrics = binary_metrics(logits[:, 0:1], target["foreground"], valid=valid)
    metrics = {f"foreground_{key}": value for key, value in metrics.items()}
    boundary_loss_proxy = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[:, 1:2],
        target["boundary"].float(),
        reduction="none",
    )
    metrics["boundary_loss"] = _safe_float(masked_mean(boundary_loss_proxy, valid))
    boundary_pred = torch.sigmoid(logits[:, 1:2]) >= 0.5
    boundary_true = target["boundary"].bool()
    if valid is not None:
        boundary_pred &= valid
        boundary_true &= valid
    boundary_intersection = (boundary_pred & boundary_true).sum().float()
    metrics["boundary_f1"] = _safe_float(
        (2 * boundary_intersection + 1e-6) / (boundary_pred.sum() + boundary_true.sum() + 1e-6)
    )
    if "distance" in target and logits.shape[1] >= 3:
        foreground = target["foreground"] > 0.5
        if valid is not None:
            foreground &= valid
        distance = torch.sigmoid(logits[:, 2:3])
        metrics["distance_mae"] = _safe_float(
            torch.abs(distance[foreground] - target["distance"][foreground]).mean()
            if torch.any(foreground)
            else distance.sum() * 0.0
        )
    if ndi is not None:
        pred = (torch.sigmoid(logits[:, 0:1]) >= 0.5).detach().cpu().numpy()
        if valid is not None:
            pred &= valid.detach().cpu().numpy()
        counts = []
        for item in pred[:, 0]:
            _, count = ndi.label(item)
            counts.append(count)
        metrics["object_count_estimate"] = float(np.mean(counts)) if counts else 0.0
    if "instances" in target and logits.shape[1] >= 3:
        p = torch.sigmoid(logits).detach().cpu().numpy()
        truths = target["instances"][:, 0].detach().cpu().numpy()
        if valid is not None:
            support = valid.detach().cpu().numpy()
            p = np.where(support, p, 0)
            truths = np.where(support[:, 0], truths, 0)
        instance_values: list[dict[str, float]] = []
        for index in range(len(truths)):
            predicted = postprocess_instance(p[index, 0], p[index, 1], p[index, 2],
                                             **({"min_object_size": 0, **(postprocessing or {})}))["labels"]
            instance_values.append(_instance_label_metrics(predicted, truths[index]))
        for key in instance_values[0] if instance_values else ():
            metrics[key] = float(np.mean([value[key] for value in instance_values]))
    return metrics


def _instance_label_metrics(prediction: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    if prediction.shape != target.shape:
        raise ValueError("Instance predictions and targets must have the same shape")
    pred_areas: dict[int, int] = {}
    true_areas: dict[int, int] = {}
    intersections: dict[tuple[int, int], int] = {}
    # One bounded contingency scan, not one whole-volume scan per object pair.
    for start in range(0, target.size, 1_048_576):
        pred = prediction.flat[start:start + 1_048_576]
        truth = target.flat[start:start + 1_048_576]
        for values, destination in ((pred, pred_areas), (truth, true_areas)):
            ids, counts = np.unique(values, return_counts=True)
            for label, count in zip(ids, counts, strict=True):
                if label:
                    destination[int(label)] = destination.get(int(label), 0) + int(count)
        support = (pred != 0) & (truth != 0)
        pairs = np.empty(np.count_nonzero(support), dtype=[("true", target.dtype), ("pred", prediction.dtype)])
        pairs["true"], pairs["pred"] = truth[support], pred[support]
        values, counts = np.unique(pairs, return_counts=True)
        for pair, count in zip(values, counts, strict=True):
            key = (int(pair["true"]), int(pair["pred"]))
            intersections[key] = intersections.get(key, 0) + int(count)
    pred_ids, true_ids = sorted(pred_areas), sorted(true_areas)
    candidates: list[tuple[float, int, int, int, int]] = []
    overlaps_by_true = dict.fromkeys(true_ids, 0)
    overlaps_by_pred = dict.fromkeys(pred_ids, 0)
    for (true_id, pred_id), intersection in intersections.items():
        union = true_areas[true_id] + pred_areas[pred_id] - intersection
        candidates.append((intersection / union, true_id, pred_id, intersection, union))
        overlaps_by_true[true_id] += 1
        overlaps_by_pred[pred_id] += 1
    matched_true: set[int] = set()
    matched_pred: set[int] = set()
    matches: list[tuple[float, int, int, int, int]] = []
    for candidate in sorted(candidates, reverse=True):
        iou, true_id, pred_id, _intersection, _union = candidate
        if iou >= threshold and true_id not in matched_true and pred_id not in matched_pred:
            matches.append(candidate)
            matched_true.add(true_id)
            matched_pred.add(pred_id)
    tp = len(matches)
    fp = len(pred_ids) - tp
    fn = len(true_ids) - tp
    denominator = tp + 0.5 * fp + 0.5 * fn
    recognition = tp / denominator if denominator else 1.0
    segmentation = float(np.mean([item[0] for item in matches])) if matches else (1.0 if not pred_ids and not true_ids else 0.0)
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    matched_intersection = sum(item[3] for item in matches)
    matched_union = sum(item[4] for item in matches)
    unmatched = sum(pred_areas[value] for value in pred_ids if value not in matched_pred)
    unmatched += sum(true_areas[value] for value in true_ids if value not in matched_true)
    return {
        "panoptic_quality": recognition * segmentation,
        "object_precision": precision,
        "object_recall": recall,
        "object_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "aggregated_jaccard": matched_intersection / max(1, matched_union + unmatched),
        "split_error_rate": sum(value > 1 for value in overlaps_by_true.values()) / max(1, len(true_ids)),
        "merge_error_rate": sum(value > 1 for value in overlaps_by_pred.values()) / max(1, len(pred_ids)),
    }


def compute_metrics(
    task: str,
    logits: Logits,
    target: torch.Tensor | dict[str, torch.Tensor],
    *, postprocessing: dict | None = None,
) -> dict[str, float]:
    logits = primary_logits(logits)
    valid = valid_support(target, logits)
    if task != "instance_friendly" and isinstance(target, dict):
        target = target["semantic"]
    if task == "binary_semantic":
        assert isinstance(target, torch.Tensor)
        prediction = None
        if postprocessing is not None:
            probabilities = torch.sigmoid(logits).detach().cpu().numpy()
            if valid is not None:
                probabilities = np.where(valid.detach().cpu().numpy(), probabilities, -1)
            masks = [postprocess_binary(p[0], threshold=postprocessing.get("threshold", 0.5),
                     min_object_size=postprocessing.get("min_object_size", 0),
                     fill_holes=postprocessing.get("fill_holes", False), connected_components=False)["mask"]
                     for p in probabilities]
            prediction = torch.from_numpy(np.stack(masks)[:, None].astype(bool)).to(logits.device)
        return binary_metrics(logits, target, valid=valid, prediction=prediction)
    if task == "multiclass_semantic":
        assert isinstance(target, torch.Tensor)
        prediction = None
        if postprocessing and postprocessing.get("min_object_size", 0) > 0:
            labels = logits.argmax(1).detach().cpu().numpy()
            if valid is not None:
                labels = np.where(valid[:, 0].detach().cpu().numpy(), labels, 0)
            for item in labels:
                for cls in range(1, logits.shape[1]):
                    region = item == cls
                    item[region & ~_remove_small(region, postprocessing["min_object_size"])] = 0
            prediction = torch.from_numpy(labels).to(logits.device)
        return multiclass_metrics(logits, target, valid=valid, prediction=prediction)
    if task == "instance_friendly":
        assert isinstance(target, dict)
        return instance_metrics(logits, target, postprocessing)
    raise ValueError(f"Unsupported task: {task}")


def primary_metric(task: str, metrics: dict[str, float]) -> float:
    if task == "binary_semantic":
        return metrics.get("dice", 0.0)
    if task == "multiclass_semantic":
        return metrics.get("mean_dice", 0.0)
    if task == "instance_friendly":
        return metrics.get("foreground_dice", 0.0)
    return 0.0
