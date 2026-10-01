"""Cheap nnU-Net-inspired 2D/3D augmentation and patch sampling."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F

from .crop_reading import CropArray
from .errors import DatasetError
from .geometry import padding_extents
from .io import normalize_image
from .scale import resize_2d_pair_to_shape, resize_3d_pair_to_shape
from .spatial_augment import SpatialImagePlan

try:  # pragma: no cover
    from scipy import ndimage as ndi
except Exception:  # pragma: no cover
    ndi = None


@dataclass(slots=True)
class AugmentationConfig:
    profile: str = "fast"
    patch_size: tuple[int, ...] = (96, 96)
    foreground_oversampling: bool = True
    foreground_probability: float = 0.4
    skip_empty_patches: bool = True
    empty_patch_max_retries: int = 8
    include_empty_patches_after_max_retries: bool = False
    instance_scale_enabled: bool = False
    target_object_diameter_px: float = 0.0
    training_scale_jitter: tuple[float, float] = (0.5, 2.0)
    min_effective_scale: float = 0.25
    max_effective_scale: float = 4.0
    flip_probability: float = 0.5
    rotate90_probability: float = 0.25
    brightness_probability: float = 0.3
    brightness_range: tuple[float, float] = (0.75, 1.25)
    shift_probability: float = 0.15
    shift_range: tuple[float, float] = (-0.1, 0.1)
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
    max_padding_ratio: float = 1.0


def make_augmentation_config(
    profile: str,
    patch_size: tuple[int, ...],
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
        if value != "auto":
            if isinstance(getattr(cfg, key), tuple):
                value = tuple(value)
            setattr(cfg, key, value)
    # A saved resolved augmentation must not override the new run's geometry plan.
    cfg.patch_size = tuple(patch_size)
    return cfg


def _pad_to_shape(image: np.ndarray, mask: np.ndarray, shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    spatial_shape = image.shape[1:]
    pads = [max(0, int(target) - int(current)) for target, current in zip(shape, spatial_shape, strict=True)]
    if all(pad == 0 for pad in pads):
        return image, mask
    spatial_pads = [(pad // 2, pad - pad // 2) for pad in pads]
    image = np.pad(image, ((0, 0), *spatial_pads), mode="constant")
    mask = np.pad(mask, tuple(spatial_pads), mode="constant")
    return image, mask


def sample_patch(
    image: np.ndarray | CropArray,
    mask: np.ndarray | CropArray,
    patch_size: tuple[int, ...],
    rng: np.random.Generator,
    foreground_oversampling: bool = True,
    foreground_probability: float = 0.4,
    center: bool = False,
    skip_empty: bool = False,
    max_retries: int = 0,
    include_empty_after_max_retries: bool = True,
    return_validity: bool = False,
    max_padding_ratio: float = 1.0,
    sampler: Any = None,
    empty: bool | None = None,
    require_background: bool = False,
) -> tuple[np.ndarray, ...]:
    pads = padding_extents(tuple(mask.shape), patch_size, max_padding_ratio)
    spatial_shape = tuple(max(length, patch) for length, patch in zip(mask.shape, patch_size, strict=True))
    for _attempt in range(max_retries + 1):
        if center:
            starts = [(dim - patch) // 2 for dim, patch in zip(spatial_shape, patch_size, strict=True)]
        elif sampler is not None:
            foreground = empty is False or (foreground_oversampling and rng.random() < foreground_probability)
            try:
                starts = sampler.starts(mask, patch_size, rng, foreground=foreground, empty=empty is True,
                                        max_padding_ratio=max_padding_ratio)
            except DatasetError as exc:
                raise EmptyPatchError(str(exc)) from exc
        elif foreground_oversampling and rng.random() < foreground_probability and np.any(np.asarray(mask) > 0):
            coordinates = np.nonzero(np.asarray(mask) > 0)
            index = int(rng.integers(len(coordinates[0])))
            starts = [
                int(np.clip(coords[index] + pad[0] - patch // 2, 0, dim - patch))
                for coords, pad, patch, dim in zip(coordinates, pads, patch_size, spatial_shape, strict=True)
            ]
        else:
            starts = [int(rng.integers(dim - patch + 1)) for dim, patch in zip(spatial_shape, patch_size, strict=True)]
        # Crop real support first; only the requested patch is padded in memory.
        source_slices = tuple(
            slice(max(0, start - pad[0]), min(length, start - pad[0] + patch))
            for start, pad, length, patch in zip(starts, pads, mask.shape, patch_size, strict=True)
        )
        patch_pads = tuple(
            (max(0, pad[0] - start), max(0, start - pad[0] + patch - length))
            for start, pad, length, patch in zip(starts, pads, mask.shape, patch_size, strict=True)
        )
        real_mask = mask[source_slices]
        positive = bool(np.any(real_mask))
        accepted = (not positive if empty is True else positive if empty is False or skip_empty else True)
        accepted = accepted and (not require_background or bool(np.any(real_mask == 0)))
        if not accepted:
            continue
        patch_image = image[(slice(None), *source_slices)]
        patch_mask = real_mask
        if any(before or after for before, after in patch_pads):
            patch_image = np.pad(patch_image, ((0, 0), *patch_pads), mode="constant")
            patch_mask = np.pad(patch_mask, patch_pads, mode="constant")
        valid = np.pad(np.ones(real_mask.shape, dtype=bool), patch_pads, mode="constant")
        result = np.ascontiguousarray(patch_image), np.ascontiguousarray(patch_mask)
        return (*result, valid) if return_validity else result
    if include_empty_after_max_retries and empty is None:
        patch_image = image[(slice(None), *source_slices)]
        patch_image = np.pad(patch_image, ((0, 0), *patch_pads), mode="constant")
        patch_mask = np.pad(real_mask, patch_pads, mode="constant")
        valid = np.pad(np.ones(real_mask.shape, dtype=bool), patch_pads, mode="constant")
        result = np.ascontiguousarray(patch_image), np.ascontiguousarray(patch_mask)
        return (*result, valid) if return_validity else result
    raise EmptyPatchError("No foreground patch found within the configured retry limit")


class EmptyPatchError(RuntimeError):
    """Signal that patch sampling should continue with another training image."""


def _spatial_affine(
    image: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
    degrees: tuple[float, float],
    scale_range: tuple[float, float],
    validity: np.ndarray | None = None,
    max_padding_ratio: float = 1.0,
    image_plan: SpatialImagePlan | None = None,
) -> tuple[np.ndarray, ...]:
    angle = math.radians(float(rng.uniform(*degrees)))
    scale = float(rng.uniform(*scale_range))
    cos_a = math.cos(angle) * scale
    sin_a = math.sin(angle) * scale
    if mask.ndim == 2:
        theta = torch.tensor([[[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0]]], dtype=torch.float32)
    elif mask.ndim == 3:
        # Rotate/scale in the high-resolution YX plane without mixing sparse Z samples.
        theta = torch.tensor(
            [[[cos_a, -sin_a, 0.0, 0.0], [sin_a, cos_a, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]],
            dtype=torch.float32,
        )
    else:
        raise ValueError("Spatial affine supports 2D and 3D arrays")
    mask_t = torch.from_numpy(np.ascontiguousarray(mask[None, None].astype(np.float32, copy=False)))
    grid = F.affine_grid(theta, [1, 1, *mask.shape], align_corners=False)
    if validity is not None:
        coords = tuple(
            ((grid[0, ..., -(axis + 1)].numpy() + 1) * mask.shape[axis] - 1) / 2 for axis in range(mask.ndim)
        )
        if not _transform_allowed(coords, validity, max_padding_ratio):
            return image, mask, validity
    if image_plan is None:
        image_t = torch.from_numpy(np.ascontiguousarray(image[None].astype(np.float32, copy=False)))
        image_out = F.grid_sample(image_t, grid, mode="bilinear", padding_mode="zeros", align_corners=False)[0].numpy()
    else:
        image_plan.affine_grid = grid
        image_out = image
    mask_out = F.grid_sample(mask_t, grid, mode="nearest", padding_mode="zeros", align_corners=False)
    result = image_out, mask_out[0, 0].numpy().astype(mask.dtype, copy=False)
    if validity is None:
        return result
    valid_t = torch.from_numpy(np.ascontiguousarray(validity[None, None], dtype=np.float32))
    valid_out = (
        F.grid_sample(valid_t, grid, mode="bilinear", padding_mode="zeros", align_corners=False)[0, 0].numpy()
        >= 1 - 1e-6
    )
    return (*result, valid_out)


def _transform_allowed(coordinates: tuple[np.ndarray, ...], validity: np.ndarray, ratio: float) -> bool:
    real = np.nonzero(validity)
    if not len(real[0]):
        return False
    for axis, coords in enumerate(coordinates):
        lo, hi = int(real[axis].min()), int(real[axis].max())
        length = hi - lo + 1
        if lo - float(coords.min()) > ratio * length or float(coords.max()) - hi > ratio * length:
            return False
    return True


def _low_resolution(
    image: np.ndarray,
    rng: np.random.Generator,
    spacing: tuple[float, ...] | None = None,
    image_plan: SpatialImagePlan | None = None,
    spatial_shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    if ndi is None:
        return image
    factor = float(rng.uniform(0.5, 0.8))
    target_shape = spatial_shape or image.shape[1:]
    axis_factors = [factor] * image[0].ndim
    if spacing is not None and len(spacing) == 3 and spacing[0] / min(spacing) >= 2:
        axis_factors[0] = 1.0
    if image_plan is not None:
        image_plan.lowres_shape = tuple(
            max(1, round(length * factor)) for length, factor in zip(target_shape, axis_factors, strict=True)
        )
        return image
    out = np.empty_like(image)
    for channel in range(image.shape[0]):
        small = ndi.zoom(image[channel], axis_factors, order=1)
        zoom = tuple(target / max(current, 1) for target, current in zip(target_shape, small.shape, strict=True))
        restored = ndi.zoom(small, zoom, order=1)
        pads = [(0, max(0, target - current)) for target, current in zip(target_shape, restored.shape, strict=True)]
        if any(pad_after > 0 for _pad_before, pad_after in pads):
            restored = np.pad(restored, pads, mode="edge")
        slices = tuple(slice(0, size) for size in target_shape)
        out[channel] = restored[slices]
    return out


def _elastic_deform(
    image: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
    spacing: tuple[float, ...] | None,
    validity: np.ndarray | None = None,
    max_padding_ratio: float = 1.0,
    image_plan: SpatialImagePlan | None = None,
) -> tuple[np.ndarray, ...]:
    if ndi is None:
        return (image, mask, validity) if validity is not None else (image, mask)
    physical_spacing = np.asarray(spacing or (1.0,) * mask.ndim, dtype=np.float64)
    physical_extent = np.asarray(mask.shape) * physical_spacing
    alpha_physical = 0.06 * float(physical_extent.min())
    sigma_physical = max(float(physical_spacing.max()), 0.08 * float(physical_extent.min()))
    displacements = []
    for axis in range(mask.ndim):
        noise = rng.normal(size=mask.shape)
        sigma_voxels = tuple(max(0.5, sigma_physical / value) for value in physical_spacing)
        smooth = ndi.gaussian_filter(noise, sigma=sigma_voxels, mode="reflect")
        maximum = max(float(np.max(np.abs(smooth))), 1e-6)
        displacements.append(smooth / maximum * (alpha_physical / physical_spacing[axis]))
    coordinates = np.meshgrid(*(np.arange(size, dtype=np.float64) for size in mask.shape), indexing="ij")
    warped = tuple(coordinates[axis] + displacements[axis] for axis in range(mask.ndim))
    if validity is not None and not _transform_allowed(warped, validity, max_padding_ratio):
        return image, mask, validity
    if image_plan is None:
        image_out = np.stack(
            [ndi.map_coordinates(channel, warped, order=1, mode="constant", cval=0) for channel in image]
        )
    else:
        normalized = [(2 * (coords + 0.5) / length - 1) for coords, length in zip(warped, mask.shape, strict=True)]
        image_plan.elastic_grid = torch.from_numpy(np.stack(normalized[::-1], axis=-1)[None].astype(np.float32))
        inside = np.logical_and.reduce(
            [(coords >= 0) & (coords <= length - 1) for coords, length in zip(warped, mask.shape, strict=True)]
        )
        image_plan.elastic_inside = torch.from_numpy(inside[None, None])
        image_out = image
    mask_out = ndi.map_coordinates(mask, warped, order=0, mode="constant", cval=0)
    result = np.ascontiguousarray(image_out), np.ascontiguousarray(mask_out.astype(mask.dtype, copy=False))
    if validity is None:
        return result
    valid_out = ndi.map_coordinates(validity.astype(np.float32), warped, order=1, mode="constant", cval=0) >= 1 - 1e-6
    return (*result, valid_out)


def apply_augmentation(
    image: np.ndarray | CropArray,
    mask: np.ndarray | CropArray,
    cfg: AugmentationConfig,
    rng: np.random.Generator | None = None,
    training: bool = True,
    object_diameter_px: float | None = None,
    spacing: tuple[float, ...] | None = None,
    normalization_statistics: dict | None = None,
    defer_photometric: bool = False,
    image_plan: SpatialImagePlan | None = None,
    return_validity: bool = False,
    sampler: Any = None,
    empty: bool | None = None,
    require_background: bool = False,
) -> tuple[np.ndarray, ...]:
    if image_plan is not None and (not training or not defer_photometric):
        raise ValueError("Deferred spatial images require training with deferred photometric augmentation")
    rng = rng or np.random.default_rng()
    if cfg.instance_scale_enabled:
        if object_diameter_px is None or object_diameter_px <= 0:
            raise ValueError("A positive object diameter estimate is required for instance scale normalization")
        jitter = 1.0
        if training:
            low, high = cfg.training_scale_jitter
            jitter = float(np.exp(rng.uniform(np.log(low), np.log(high))))
        scale = float(
            np.clip(
                cfg.target_object_diameter_px / object_diameter_px * jitter,
                cfg.min_effective_scale,
                cfg.max_effective_scale,
            )
        )
        source_patch_size = tuple(max(1, int(round(size / scale))) for size in cfg.patch_size)
        try:
            padding_extents(tuple(mask.shape), source_patch_size, cfg.max_padding_ratio)
        except DatasetError:
            # Clamp infeasible jitter to the smallest feasible scale, keeping shape fixed.
            feasible_scale = max(
                p / max(1, math.floor((1 + 2 * cfg.max_padding_ratio) * length))
                for p, length in zip(cfg.patch_size, mask.shape, strict=True)
            )
            scale = max(scale, feasible_scale)
            if scale > cfg.max_effective_scale:
                raise EmptyPatchError("No feasible scale within the padding/scale limits") from None
            source_patch_size = tuple(max(1, int(round(size / scale))) for size in cfg.patch_size)
        image, mask, validity = sample_patch(
            image,
            mask,
            source_patch_size,
            rng,
            foreground_oversampling=cfg.foreground_oversampling,
            foreground_probability=cfg.foreground_probability,
            center=not training and sampler is None,
            skip_empty=training and cfg.skip_empty_patches,
            max_retries=cfg.empty_patch_max_retries if training and cfg.skip_empty_patches else 0,
            include_empty_after_max_retries=cfg.include_empty_patches_after_max_retries,
            return_validity=True,
            max_padding_ratio=cfg.max_padding_ratio,
            sampler=sampler, empty=empty, require_background=require_background,
        )
        if normalization_statistics is not None:
            image = normalize_image(image, statistics=normalization_statistics)
            image[:, ~validity] = 0.0
        valid_t = torch.from_numpy(np.ascontiguousarray(validity[None, None], dtype=np.float32))
        validity = (
            F.interpolate(
                valid_t, size=cfg.patch_size, mode="bilinear" if mask.ndim == 2 else "trilinear", align_corners=False
            )[0, 0].numpy()
            >= 1 - 1e-6
        )
        if image_plan is not None:
            image_plan.resize_shape = cfg.patch_size
            indices = [
                np.minimum(np.arange(target) * source // target, source - 1)
                for source, target in zip(mask.shape, cfg.patch_size, strict=True)
            ]
            mask = np.ascontiguousarray(mask[np.ix_(*indices)])
        elif mask.ndim == 2:
            image, mask = resize_2d_pair_to_shape(image, mask, cast(tuple[int, int], cfg.patch_size))
        elif mask.ndim == 3:
            image, mask = resize_3d_pair_to_shape(image, mask, cast(tuple[int, int, int], cfg.patch_size))
        else:
            raise ValueError("Instance scale normalization supports 2D and 3D masks")
    else:
        image, mask, validity = sample_patch(
            image,
            mask,
            cfg.patch_size,
            rng,
            foreground_oversampling=cfg.foreground_oversampling,
            foreground_probability=cfg.foreground_probability,
            center=not training and sampler is None,
            skip_empty=training and cfg.skip_empty_patches,
            max_retries=cfg.empty_patch_max_retries if training and cfg.skip_empty_patches else 0,
            include_empty_after_max_retries=cfg.include_empty_patches_after_max_retries,
            return_validity=True,
            max_padding_ratio=cfg.max_padding_ratio,
            sampler=sampler, empty=empty, require_background=require_background,
        )
        if normalization_statistics is not None:
            image = normalize_image(image, statistics=normalization_statistics)
            image[:, ~validity] = 0.0
    if not image.flags.writeable:
        image = image.copy()
    if not training:
        image[:, ~validity] = 0.0
        result = image.astype(np.float32, copy=False), mask.astype(np.int64, copy=False)
        return (*result, validity) if return_validity else result

    for spatial_axis in range(mask.ndim):
        if rng.random() < cfg.flip_probability:
            if image_plan is None:
                image = np.flip(image, axis=spatial_axis + 1)
            else:
                image_plan.flips.append(spatial_axis)
            mask = np.flip(mask, axis=spatial_axis)
            validity = np.flip(validity, axis=spatial_axis)
    if rng.random() < cfg.rotate90_probability and mask.shape[-2] == mask.shape[-1]:
        k = int(rng.integers(0, 4))
        if image_plan is None:
            image = np.rot90(image, k, axes=(-2, -1))
        else:
            image_plan.quarter_turns = k
        mask = np.rot90(mask, k, axes=(-2, -1))
        validity = np.rot90(validity, k, axes=(-2, -1))
    if cfg.affine_probability > 0 and rng.random() < cfg.affine_probability:
        image, mask, validity = _spatial_affine(
            image, mask, rng, cfg.rotation_degrees, cfg.scale_range, validity, cfg.max_padding_ratio, image_plan
        )
    if cfg.lowres_probability > 0 and rng.random() < cfg.lowres_probability:
        image = _low_resolution(image, rng, spacing, image_plan, tuple(mask.shape))
    if cfg.elastic_probability > 0 and rng.random() < cfg.elastic_probability:
        image, mask, validity = _elastic_deform(image, mask, rng, spacing, validity, cfg.max_padding_ratio, image_plan)
    if not np.any(validity):
        raise EmptyPatchError("Transform left no real supervised support")
    if return_validity and cfg.skip_empty_patches and not np.any(mask[validity] > 0):
        raise EmptyPatchError("Transform left no foreground in real support")
    if empty is not None and bool(np.any(mask[validity])) == empty:
        raise EmptyPatchError("Transform changed the scheduled foreground/empty patch category")
    if require_background and not np.any(mask[validity] == 0):
        raise EmptyPatchError("Transform left no real background")
    if cfg.blur_probability > 0 and rng.random() < cfg.blur_probability and ndi is not None:
        sigma = float(rng.uniform(*cfg.blur_sigma))
        # Do not over-blur a coarse Z axis; in-plane filtering remains isotropic.
        per_axis_sigma = (min(0.5, sigma), sigma, sigma) if mask.ndim == 3 else (sigma, sigma)
        if image_plan is None:
            for channel in range(image.shape[0]):
                image[channel] = ndi.gaussian_filter(image[channel], sigma=per_axis_sigma)
        else:
            image_plan.blur_sigmas = per_axis_sigma
    if not defer_photometric:
        if rng.random() < cfg.brightness_probability:
            image = np.asarray(image * float(rng.uniform(*cfg.brightness_range)))
        if rng.random() < cfg.shift_probability:
            image = np.asarray(image + float(rng.uniform(*cfg.shift_range)))
        if rng.random() < cfg.contrast_probability:
            mean = image[:, validity].mean(axis=1).reshape((-1,) + (1,) * mask.ndim)
            image = np.asarray((image - mean) * float(rng.uniform(*cfg.contrast_range)) + mean)
        if rng.random() < cfg.gamma_probability:
            gamma = float(rng.uniform(*cfg.gamma_range))
            min_value = image[:, validity].min(axis=1).reshape((-1,) + (1,) * mask.ndim)
            max_value = image[:, validity].max(axis=1).reshape((-1,) + (1,) * mask.ndim)
            denom = np.maximum(max_value - min_value, 1e-6)
            normalized = np.clip((image - min_value) / denom, 0.0, 1.0)
            image = np.asarray(np.power(normalized, gamma) * denom + min_value)
        if rng.random() < cfg.noise_probability:
            image = np.asarray(image + rng.normal(0.0, cfg.noise_std, size=image.shape).astype(np.float32))
        if image.shape[0] > 1 and rng.random() < cfg.channel_dropout_probability:
            channel = int(rng.integers(0, image.shape[0]))
            image[channel] = 0
    if image_plan is None:
        image[:, ~validity] = 0.0
    result = np.ascontiguousarray(image.astype(np.float32, copy=False)), np.ascontiguousarray(mask.astype(np.int64))
    return (*result, np.ascontiguousarray(validity)) if return_validity else result


def apply_tensor_photometric_augmentation(
    images: torch.Tensor,
    valid: torch.Tensor | None,
    cfg: AugmentationConfig,
) -> torch.Tensor:
    """Apply image-only random augmentations on BCHW or BCDHW tensors."""
    if images.ndim not in {4, 5}:
        raise ValueError(f"Expected BCHW or BCDHW input, got shape {tuple(images.shape)}")

    batch, channels = images.shape[:2]
    device, dtype = images.device, images.dtype
    spatial_dims = tuple(range(2, images.ndim))
    scalar_shape = (batch, 1, *([1] * (images.ndim - 2)))

    def sample_probability(probability: float) -> torch.Tensor:
        return (torch.rand(batch, device=device) < probability).view(scalar_shape)

    def sample_uniform(bounds: tuple[float, float]) -> torch.Tensor:
        low, high = bounds
        return (low + (high - low) * torch.rand(batch, device=device, dtype=dtype)).view(scalar_shape)

    if cfg.brightness_probability > 0:
        apply = sample_probability(cfg.brightness_probability)
        images = torch.where(apply, images * sample_uniform(cfg.brightness_range), images)

    if cfg.shift_probability > 0:
        apply = sample_probability(cfg.shift_probability)
        images = torch.where(apply, images + sample_uniform(cfg.shift_range), images)

    valid_bool = valid.bool() if valid is not None else None
    valid_float = valid_bool.to(dtype=dtype) if valid_bool is not None else None

    if cfg.contrast_probability > 0:
        apply = sample_probability(cfg.contrast_probability)
        factor = sample_uniform(cfg.contrast_range)
        if valid_float is None:
            mean = images.mean(dim=spatial_dims, keepdim=True)
        else:
            denominator = valid_float.sum(dim=spatial_dims, keepdim=True).clamp_min(1.0)
            mean = (images * valid_float).sum(dim=spatial_dims, keepdim=True) / denominator
        images = torch.where(apply, (images - mean) * factor + mean, images)

    if cfg.gamma_probability > 0:
        apply = sample_probability(cfg.gamma_probability)
        gamma = sample_uniform(cfg.gamma_range)
        if valid_bool is None:
            minimum = images.amin(dim=spatial_dims, keepdim=True)
            maximum = images.amax(dim=spatial_dims, keepdim=True)
        else:
            support = valid_bool.expand(-1, channels, *([-1] * (images.ndim - 2)))
            minimum = images.masked_fill(~support, float("inf")).amin(dim=spatial_dims, keepdim=True)
            maximum = images.masked_fill(~support, float("-inf")).amax(dim=spatial_dims, keepdim=True)
        scale = (maximum - minimum).clamp_min(1e-6)
        adjusted = ((images - minimum) / scale).clamp(0.0, 1.0).pow(gamma) * scale + minimum
        images = torch.where(apply, adjusted, images)

    if cfg.noise_probability > 0:
        apply = sample_probability(cfg.noise_probability).to(dtype=dtype)
        images = images + torch.randn_like(images) * cfg.noise_std * apply

    if channels > 1 and cfg.channel_dropout_probability > 0:
        selected = (torch.rand(batch, device=device) < cfg.channel_dropout_probability).to(dtype=dtype)
        selected_channel = torch.randint(channels, (batch, 1), device=device)
        drop = torch.zeros(batch, channels, device=device, dtype=dtype)
        drop.scatter_(1, selected_channel, selected[:, None])
        images = images * (1.0 - drop).view(batch, channels, *([1] * (images.ndim - 2)))

    return images.masked_fill(~valid.bool(), 0.0) if valid is not None else images
