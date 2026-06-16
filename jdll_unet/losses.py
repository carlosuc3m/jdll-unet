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


def focal_binary_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    alpha: float | None = None,
) -> torch.Tensor:
    target = target.float()
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    probs = torch.sigmoid(logits)
    p_t = probs * target + (1 - probs) * (1 - target)
    focal = bce * (1 - p_t).pow(gamma)
    if alpha is not None:
        alpha_t = alpha * target + (1 - alpha) * (1 - target)
        focal = alpha_t * focal
    return focal.mean()


def multiclass_focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    alpha: float | None = None,
) -> torch.Tensor:
    target = target.long()
    ce = F.cross_entropy(logits, target, reduction="none")
    p_t = torch.exp(-ce)
    focal = ce * (1 - p_t).pow(gamma)
    if alpha is not None:
        alpha_t = torch.where(target > 0, torch.full_like(focal, alpha), torch.full_like(focal, 1 - alpha))
        focal = alpha_t * focal
    return focal.mean()


def compute_loss(
    task: str,
    logits: Logits,
    target: Target,
    weights: dict[str, float] | None = None,
    focal_gamma: float = 2.0,
    focal_alpha: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if isinstance(logits, (list, tuple)):
        primary_loss, components = _compute_single_loss(task, logits[0], target, weights, focal_gamma, focal_alpha)
        aux_losses = []
        for index, aux_logits in enumerate(logits[1:]):
            aux_target = resize_target_for_logits(task, target, aux_logits)
            aux_loss, _aux_components = _compute_single_loss(task, aux_logits, aux_target, weights, focal_gamma, focal_alpha)
            aux_losses.append((0.5 ** (index + 1), aux_loss))
        if not aux_losses:
            return primary_loss, components
        total_weight = 1.0 + sum(weight for weight, _loss in aux_losses)
        aux_weighted = sum(weight * aux_loss for weight, aux_loss in aux_losses)
        total = (primary_loss + aux_weighted) / total_weight
        components["deep_supervision_loss"] = (aux_weighted / (total_weight - 1.0)).detach()
        return total, components
    return _compute_single_loss(task, logits, target, weights, focal_gamma, focal_alpha)


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
    focal_gamma: float = 2.0,
    focal_alpha: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    weights = weights or {}
    if task == "binary_semantic":
        assert isinstance(target, torch.Tensor)
        bce = F.binary_cross_entropy_with_logits(logits, target.float())
        dice = binary_dice_loss(logits, target)
        total = weights.get("bce", 1.0) * bce + weights.get("dice", 1.0) * dice
        components = {"bce_loss": bce.detach(), "dice_loss": dice.detach()}
        if weights.get("focal", 0.0) > 0:
            focal = focal_binary_loss(logits, target, gamma=focal_gamma, alpha=focal_alpha)
            total = total + weights.get("focal", 0.0) * focal
            components["focal_loss"] = focal.detach()
        return total, components
    if task == "multiclass_semantic":
        assert isinstance(target, torch.Tensor)
        ce = F.cross_entropy(logits, target.long())
        dice = multiclass_dice_loss(logits, target)
        total = weights.get("cross_entropy", 1.0) * ce + weights.get("dice", 1.0) * dice
        components = {"cross_entropy_loss": ce.detach(), "dice_loss": dice.detach()}
        if weights.get("focal", 0.0) > 0:
            focal = multiclass_focal_loss(logits, target, gamma=focal_gamma, alpha=focal_alpha)
            total = total + weights.get("focal", 0.0) * focal
            components["focal_loss"] = focal.detach()
        return total, components
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
        components = {
            "foreground_bce_loss": fg_bce.detach(),
            "foreground_dice_loss": fg_dice.detach(),
            "boundary_loss": boundary_loss.detach(),
        }
        if weights.get("focal", 0.0) > 0:
            fg_focal = focal_binary_loss(fg_logits, foreground, gamma=focal_gamma, alpha=focal_alpha)
            total = total + weights.get("focal", 0.0) * fg_focal
            components["foreground_focal_loss"] = fg_focal.detach()
        if weights.get("boundary_focal", 0.0) > 0:
            boundary_focal = focal_binary_loss(boundary_logits, boundary, gamma=focal_gamma, alpha=focal_alpha)
            total = total + weights.get("boundary_focal", 0.0) * boundary_focal
            components["boundary_focal_loss"] = boundary_focal.detach()
        return total, components
    raise ValueError(f"Unsupported task: {task}")
