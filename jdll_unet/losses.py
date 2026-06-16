"""Automatic losses for JDLL UNet tasks."""

from __future__ import annotations

import torch
import torch.nn.functional as F

Logits = torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...]
Target = torch.Tensor | dict[str, torch.Tensor]


def primary_logits(logits: Logits) -> torch.Tensor:
    return logits[0] if isinstance(logits, (list, tuple)) else logits


def binary_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    target = target.float()
    dims = tuple(range(1, probs.ndim))
    intersection = (probs * target).sum(dim=dims)
    denom = probs.sum(dim=dims) + target.sum(dim=dims)
    dice = (2 * intersection + eps) / (denom + eps)
    return 1.0 - dice.mean()


def multiclass_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    classes = logits.shape[1]
    one_hot = F.one_hot(target.long().clamp_min(0), num_classes=classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    intersection = (probs * one_hot).sum(dim=dims)
    denom = probs.sum(dim=dims) + one_hot.sum(dim=dims)
    dice = (2 * intersection + eps) / (denom + eps)
    if classes > 1:
        dice = dice[1:]
    return 1.0 - dice.mean()


def focal_binary_loss(logits: torch.Tensor, target: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    target = target.float()
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    probs = torch.sigmoid(logits)
    p_t = probs * target + (1 - probs) * (1 - target)
    return (bce * (1 - p_t).pow(gamma)).mean()


def compute_loss(
    task: str,
    logits: Logits,
    target: Target,
    weights: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if isinstance(logits, (list, tuple)):
        primary_loss, components = _compute_single_loss(task, logits[0], target, weights)
        aux_losses = []
        for index, aux_logits in enumerate(logits[1:]):
            aux_target = resize_target_for_logits(task, target, aux_logits)
            aux_loss, _aux_components = _compute_single_loss(task, aux_logits, aux_target, weights)
            aux_losses.append((0.5 ** (index + 1), aux_loss))
        if not aux_losses:
            return primary_loss, components
        total_weight = 1.0 + sum(weight for weight, _loss in aux_losses)
        aux_weighted = sum(weight * aux_loss for weight, aux_loss in aux_losses)
        total = (primary_loss + aux_weighted) / total_weight
        components["deep_supervision_loss"] = (aux_weighted / (total_weight - 1.0)).detach()
        return total, components
    return _compute_single_loss(task, logits, target, weights)


def resize_target_for_logits(task: str, target: Target, logits: torch.Tensor) -> Target:
    size = logits.shape[-2:]
    if task == "multiclass_semantic":
        assert isinstance(target, torch.Tensor)
        return F.interpolate(target[:, None].float(), size=size, mode="nearest")[:, 0].long()
    if isinstance(target, dict):
        return {key: F.interpolate(value.float(), size=size, mode="nearest") for key, value in target.items()}
    return F.interpolate(target.float(), size=size, mode="nearest")


def _compute_single_loss(
    task: str,
    logits: torch.Tensor,
    target: Target,
    weights: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    weights = weights or {}
    if task == "binary_semantic":
        assert isinstance(target, torch.Tensor)
        bce = F.binary_cross_entropy_with_logits(logits, target.float())
        dice = binary_dice_loss(logits, target)
        total = weights.get("bce", 1.0) * bce + weights.get("dice", 1.0) * dice
        return total, {"bce_loss": bce.detach(), "dice_loss": dice.detach()}
    if task == "multiclass_semantic":
        assert isinstance(target, torch.Tensor)
        ce = F.cross_entropy(logits, target.long())
        dice = multiclass_dice_loss(logits, target)
        total = weights.get("cross_entropy", 1.0) * ce + weights.get("dice", 1.0) * dice
        return total, {"cross_entropy_loss": ce.detach(), "dice_loss": dice.detach()}
    if task == "instance_friendly":
        assert isinstance(target, dict)
        foreground = target["foreground"].float()
        boundary = target["boundary"].float()
        fg_logits = logits[:, 0:1]
        boundary_logits = logits[:, 1:2]
        fg_bce = F.binary_cross_entropy_with_logits(fg_logits, foreground)
        fg_dice = binary_dice_loss(fg_logits, foreground)
        boundary_loss = F.binary_cross_entropy_with_logits(boundary_logits, boundary)
        total = (
            weights.get("bce", 1.0) * fg_bce
            + weights.get("dice", 1.0) * fg_dice
            + weights.get("boundary", 0.5) * boundary_loss
        )
        return total, {
            "foreground_bce_loss": fg_bce.detach(),
            "foreground_dice_loss": fg_dice.detach(),
            "boundary_loss": boundary_loss.detach(),
        }
    raise ValueError(f"Unsupported task: {task}")
