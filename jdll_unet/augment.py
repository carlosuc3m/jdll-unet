"""Cheap nnU-Net-inspired 2D augmentation and patch sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math

import numpy as np
import torch
import torch.nn.functional as F

try:  # pragma: no cover
    from scipy import ndimage as ndi
except Exception:  # pragma: no cover
    ndi = None


@dataclass(slots=True)
class AugmentationConfig:
    profile: str = "fast"
    patch_size: tuple[int, int] = (96, 96)
    foreground_oversampling: bool = True
    foreground_probability: float = 0.4
    flip_probability: float = 0.5
    rotate90_probability: float = 0.25
    brightness_probability: float = 0.3
    brightness_range: tuple[float, float] = (0.75, 1.25)
    contrast_probability: float = 0.3
    contrast_range: tuple[float, float] = (0.75, 1.25)
    gamma_probability: float = 0.2
    gamma_range: tuple[float, float] = (0.7, 1.5)
    noise_probability: float = 0.15
    noise_std: float = 0.03
    blur_probability: float = 0.1
    blur_sigma: tuple[float, float] = (0.5, 1.0)
    channel_dropout_probability: float = 0.05
    affine_probability: float = 0.0
    rotation_degrees: tuple[float, float] = (-15.0, 15.0)
    scale_range: tuple[float, float] = (0.85, 1.25)
    lowres_probability: float = 0.0
    elastic_probability: float = 0.0


def make_augmentation_config(
    profile: str,
    patch_size: tuple[int, int],
    foreground_oversampling: bool,
    foreground_probability: float,
    overrides: dict[str, Any] | None = None,
) -> AugmentationConfig:
    cfg = AugmentationConfig(
        profile=profile,
        patch_size=patch_size,
        foreground_oversampling=foreground_oversampling,
        foreground_probability=foreground_probability,
    )
    if profile in {"light-balanced", "balanced", "strong"}:
        cfg.affine_probability = 0.25
        cfg.blur_probability = 0.15
        cfg.lowres_probability = 0.1
    if profile == "balanced":
        cfg.affine_probability = 0.35
        cfg.noise_probability = 0.2
    if profile == "strong":
        cfg.affine_probability = 0.5
        cfg.elastic_probability = 0.15
        cfg.rotation_degrees = (-35.0, 35.0)
        cfg.scale_range = (0.7, 1.4)
        cfg.blur_probability = 0.25
        cfg.lowres_probability = 0.2
    for key, value in (overrides or {}).items():
        if not hasattr(cfg, key):
            raise ValueError(f"Unknown augmentation parameter: {key}")
        setattr(cfg, key, value)
    return cfg


def _pad_to_shape(image: np.ndarray, mask: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[-2:]
    pad_y = max(0, shape[0] - height)
    pad_x = max(0, shape[1] - width)
    if pad_y == 0 and pad_x == 0:
        return image, mask
    before_y = pad_y // 2
    after_y = pad_y - before_y
    before_x = pad_x // 2
    after_x = pad_x - before_x
    image = np.pad(image, ((0, 0), (before_y, after_y), (before_x, after_x)), mode="reflect")
    mask = np.pad(mask, ((before_y, after_y), (before_x, after_x)), mode="constant")
    return image, mask


def sample_patch(
    image: np.ndarray,
    mask: np.ndarray,
    patch_size: tuple[int, int],
    rng: np.random.Generator,
    foreground_oversampling: bool = True,
    foreground_probability: float = 0.4,
    center: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    image, mask = _pad_to_shape(image, mask, patch_size)
    height, width = image.shape[-2:]
    patch_h, patch_w = patch_size
    if center:
        y0 = max(0, (height - patch_h) // 2)
        x0 = max(0, (width - patch_w) // 2)
    elif foreground_oversampling and rng.random() < foreground_probability and np.any(mask != 0):
        ys, xs = np.nonzero(mask != 0)
        index = int(rng.integers(0, len(ys)))
        cy, cx = int(ys[index]), int(xs[index])
        y0 = int(np.clip(cy - patch_h // 2, 0, max(0, height - patch_h)))
        x0 = int(np.clip(cx - patch_w // 2, 0, max(0, width - patch_w)))
    else:
        y0 = int(rng.integers(0, max(1, height - patch_h + 1)))
        x0 = int(rng.integers(0, max(1, width - patch_w + 1)))
    patch_img = image[:, y0 : y0 + patch_h, x0 : x0 + patch_w]
    patch_mask = mask[y0 : y0 + patch_h, x0 : x0 + patch_w]
    return np.ascontiguousarray(patch_img), np.ascontiguousarray(patch_mask)


def _spatial_affine(
    image: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
    degrees: tuple[float, float],
    scale_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    angle = math.radians(float(rng.uniform(*degrees)))
    scale = float(rng.uniform(*scale_range))
    cos_a = math.cos(angle) * scale
    sin_a = math.sin(angle) * scale
    theta = torch.tensor([[[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0]]], dtype=torch.float32)
    image_t = torch.from_numpy(np.ascontiguousarray(image[None].astype(np.float32, copy=False)))
    mask_t = torch.from_numpy(np.ascontiguousarray(mask[None, None].astype(np.float32, copy=False)))
    grid = F.affine_grid(theta, image_t.shape, align_corners=False)
    image_out = F.grid_sample(image_t, grid, mode="bilinear", padding_mode="border", align_corners=False)
    mask_out = F.grid_sample(mask_t, grid, mode="nearest", padding_mode="zeros", align_corners=False)
    return image_out[0].numpy(), mask_out[0, 0].numpy().astype(mask.dtype, copy=False)


def _low_resolution(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if ndi is None:
        return image
    factor = float(rng.uniform(0.5, 0.8))
    out = np.empty_like(image)
    for channel in range(image.shape[0]):
        small = ndi.zoom(image[channel], factor, order=1)
        zoom = (image.shape[1] / small.shape[0], image.shape[2] / small.shape[1])
        restored = ndi.zoom(small, zoom, order=1)
        out[channel] = restored[: image.shape[1], : image.shape[2]]
    return out


def apply_augmentation(
    image: np.ndarray,
    mask: np.ndarray,
    cfg: AugmentationConfig,
    rng: np.random.Generator | None = None,
    training: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    rng = rng or np.random.default_rng()
    image, mask = sample_patch(
        image,
        mask,
        cfg.patch_size,
        rng,
        foreground_oversampling=cfg.foreground_oversampling,
        foreground_probability=cfg.foreground_probability,
        center=not training,
    )
    if not training:
        return image.astype(np.float32, copy=False), mask.astype(np.int64, copy=False)

    if rng.random() < cfg.flip_probability:
        image = image[..., ::-1]
        mask = mask[..., ::-1]
    if rng.random() < cfg.flip_probability:
        image = image[..., ::-1, :]
        mask = mask[::-1, :]
    if rng.random() < cfg.rotate90_probability and image.shape[-2] == image.shape[-1]:
        k = int(rng.integers(0, 4))
        image = np.rot90(image, k, axes=(-2, -1))
        mask = np.rot90(mask, k, axes=(-2, -1))
    if cfg.affine_probability > 0 and rng.random() < cfg.affine_probability:
        image, mask = _spatial_affine(image, mask, rng, cfg.rotation_degrees, cfg.scale_range)
    if cfg.lowres_probability > 0 and rng.random() < cfg.lowres_probability:
        image = _low_resolution(image, rng)
    if cfg.blur_probability > 0 and rng.random() < cfg.blur_probability and ndi is not None:
        sigma = float(rng.uniform(*cfg.blur_sigma))
        for channel in range(image.shape[0]):
            image[channel] = ndi.gaussian_filter(image[channel], sigma=sigma)
    if rng.random() < cfg.brightness_probability:
        image = image * float(rng.uniform(*cfg.brightness_range))
    if rng.random() < cfg.contrast_probability:
        mean = image.mean(axis=(-2, -1), keepdims=True)
        image = (image - mean) * float(rng.uniform(*cfg.contrast_range)) + mean
    if rng.random() < cfg.gamma_probability:
        gamma = float(rng.uniform(*cfg.gamma_range))
        min_value = image.min(axis=(-2, -1), keepdims=True)
        max_value = image.max(axis=(-2, -1), keepdims=True)
        denom = np.maximum(max_value - min_value, 1e-6)
        normalized = np.clip((image - min_value) / denom, 0.0, 1.0)
        image = np.power(normalized, gamma) * denom + min_value
    if rng.random() < cfg.noise_probability:
        image = image + rng.normal(0.0, cfg.noise_std, size=image.shape).astype(np.float32)
    if image.shape[0] > 1 and rng.random() < cfg.channel_dropout_probability:
        channel = int(rng.integers(0, image.shape[0]))
        image[channel] = 0
    return np.ascontiguousarray(image.astype(np.float32, copy=False)), np.ascontiguousarray(mask.astype(np.int64))
