"""Replay image transforms on the training device using CPU-planned label geometry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import default_collate


@dataclass
class SpatialImagePlan:
    resize_shape: tuple[int, ...] | None = None
    flips: list[int] = field(default_factory=list)
    quarter_turns: int = 0
    affine_grid: torch.Tensor | None = None
    lowres_shape: tuple[int, ...] | None = None
    elastic_grid: torch.Tensor | None = None
    elastic_inside: torch.Tensor | None = None
    blur_sigmas: tuple[float, ...] | None = None


@dataclass
class SpatialImageSample:
    image: torch.Tensor
    plan: SpatialImagePlan


@dataclass
class SpatialImageBatch:
    images: torch.Tensor | list[torch.Tensor]
    plans: list[SpatialImagePlan]

    def pin_memory(self) -> SpatialImageBatch:
        # Ragged crops and grids are packed and pinned once per compatible group.
        images = self.images.pin_memory() if isinstance(self.images, torch.Tensor) else self.images
        return SpatialImageBatch(images, self.plans)

    def materialize(self, device: torch.device) -> torch.Tensor:
        if isinstance(self.images, torch.Tensor):
            images = self.images.to(device, non_blocking=True)
            images = _resize_image_group(images, self.plans[0].resize_shape)
        else:
            # Group variable-size crops rather than padding an entire 3D batch
            # to its largest source crop. Peak device memory stays bounded.
            groups: dict[tuple[int, ...], list[int]] = {}
            for index, image in enumerate(self.images):
                groups.setdefault(tuple(image.shape), []).append(index)
            output_shape = self.plans[0].resize_shape
            assert output_shape is not None
            images = torch.empty(
                (len(self.images), self.images[0].shape[0], *output_shape), device=device, dtype=self.images[0].dtype
            )
            for indices in groups.values():
                group = _stack_cpu_tensors([self.images[index] for index in indices])
                if device.type == "cuda":
                    group = group.pin_memory()
                resized = _resize_image_group(group.to(device, non_blocking=True), output_shape)
                images.index_copy_(0, torch.tensor(indices, device=device), resized)
        return apply_spatial_image_plans(images, self.plans)


def _stack_cpu_tensors(tensors: list[torch.Tensor]) -> torch.Tensor:
    # Small host copies should not wake a large intra-op pool before CUDA dispatch.
    if len(tensors) == 1:
        return tensors[0].unsqueeze(0)
    return torch.from_numpy(np.stack([tensor.numpy() for tensor in tensors]))


def collate_spatial_samples(samples: list[tuple[SpatialImageSample, Any]]) -> tuple[SpatialImageBatch, Any]:
    images, targets = zip(*samples, strict=True)
    shapes = {sample.plan.resize_shape or tuple(sample.image.shape[1:]) for sample in images}
    if len(shapes) != 1:
        raise ValueError("A spatial batch requires one fixed output patch shape")
    tensors = [sample.image for sample in images]
    packed = _stack_cpu_tensors(tensors) if len({tuple(image.shape) for image in tensors}) == 1 else tensors
    if isinstance(targets[0], dict):
        collated = {key: _stack_cpu_tensors([target[key] for target in targets]) for key in targets[0]}
    else:
        collated = default_collate(list(targets))
    return (
        SpatialImageBatch(packed, [sample.plan for sample in images]),
        collated,
    )


def _gaussian_blur(images: torch.Tensor, sigmas: tuple[float, ...]) -> torch.Tensor:
    return _batch_gaussian_blur(images, [sigmas] * images.shape[0])


def _batch_gaussian_blur(images: torch.Tensor, sigmas: list[tuple[float, ...]]) -> torch.Tensor:
    """Separable filtering with SciPy's half-sample symmetric boundary convention."""
    spatial_dims = images.ndim - 2
    convolution = F.conv3d if spatial_dims == 3 else F.conv2d
    batch, channels = images.shape[:2]
    for axis in range(spatial_dims):
        values = [sample[axis] for sample in sigmas]
        radii = [int(4.0 * sigma + 0.5) for sigma in values]
        radius = max(radii)
        if radius == 0:
            continue
        offsets = torch.arange(-radius, radius + 1, device=images.device, dtype=images.dtype)
        sigma_t = torch.tensor(values, device=images.device, dtype=images.dtype)[:, None].clamp_min(1e-6)
        radius_t = torch.tensor(radii, device=images.device)[:, None]
        kernel = torch.exp(-0.5 * (offsets[None] / sigma_t).square()) * (offsets.abs()[None] <= radius_t)
        kernel = kernel / kernel.sum(1, keepdim=True)
        shape = [1] * spatial_dims
        shape[axis] = offsets.numel()
        weight = kernel[:, None].expand(batch, channels, -1).reshape(batch * channels, 1, *shape).contiguous()
        length = images.shape[axis + 2]
        indices = torch.arange(-radius, length + radius, device=images.device).remainder(2 * length)
        indices = torch.where(indices < length, indices, 2 * length - 1 - indices)
        padded = images.index_select(axis + 2, indices).flatten(0, 1)[None]
        images = convolution(padded, weight, groups=batch * channels).reshape(batch, channels, *images.shape[2:])
    return images


def _resize_image_group(images: torch.Tensor, shape: tuple[int, ...] | None) -> torch.Tensor:
    if shape is None or tuple(images.shape[2:]) == shape:
        return images
    return F.interpolate(images, size=shape, mode="trilinear" if images.ndim == 5 else "bilinear", align_corners=False)


def _orient_batch(images: torch.Tensor, plans: list[SpatialImagePlan]) -> torch.Tensor:
    if not any(plan.flips or plan.quarter_turns for plan in plans):
        return images
    spatial = tuple(images.shape[2:])
    batch = len(plans)
    scalar = (batch, *([1] * len(spatial)))
    coords = list(torch.meshgrid(*(torch.arange(n, device=images.device) for n in spatial), indexing="ij"))
    coords = [value[None].expand(batch, *spatial) for value in coords]
    y, x = coords[-2:]
    turns = torch.tensor([plan.quarter_turns for plan in plans], device=images.device).reshape(scalar)
    coords[-2] = torch.where(
        turns == 1, x, torch.where(turns == 2, spatial[-2] - 1 - y, torch.where(turns == 3, spatial[-2] - 1 - x, y))
    )
    coords[-1] = torch.where(
        turns == 1, spatial[-1] - 1 - y, torch.where(turns == 2, spatial[-1] - 1 - x, torch.where(turns == 3, y, x))
    )
    flat_index = torch.zeros_like(coords[0])
    for axis, (length, coord) in enumerate(zip(spatial, coords, strict=True)):
        flip = torch.tensor([axis in plan.flips for plan in plans], device=images.device).reshape(scalar)
        flat_index = flat_index * length + torch.where(flip, length - 1 - coord, coord)
    index = flat_index.flatten(1)[:, None].expand(-1, images.shape[1], -1)
    return images.flatten(2).gather(2, index).reshape_as(images)


def _batch_warp(images: torch.Tensor, plans: list[SpatialImagePlan], field: str) -> torch.Tensor:
    indices = [index for index, plan in enumerate(plans) if getattr(plan, field) is not None]
    if not indices:
        return images
    grids = torch.cat([getattr(plans[index], field) for index in indices])
    if images.is_cuda:
        grids = grids.pin_memory()
    selection = torch.tensor(indices, device=images.device)
    selected = images if len(indices) == len(plans) else images.index_select(0, selection)
    warped = F.grid_sample(
        selected, grids.to(images.device, non_blocking=True), mode="bilinear", padding_mode="zeros", align_corners=False
    )
    if field == "elastic_grid":
        supports = [plans[index].elastic_inside for index in indices]
        assert all(support is not None for support in supports)
        inside = torch.cat([support for support in supports if support is not None])
        if images.is_cuda:
            inside = inside.pin_memory()
        warped = warped.masked_fill(~inside.to(images.device, non_blocking=True), 0.0)
    return warped if len(indices) == len(plans) else images.index_copy(0, selection, warped)


def _resize_grid(
    source_shapes: list[tuple[int, ...]],
    target_shapes: list[tuple[int, ...]],
    canvas: tuple[int, ...],
    output: tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    """Align-corners grids for a batch of differently sized low-resolution images."""
    batch, ndim = len(source_shapes), len(canvas)
    vectors = []
    for axis, length in enumerate(output):
        view = [1] * (ndim + 1)
        view[axis + 1] = length
        position = torch.arange(length, device=device, dtype=torch.float32).reshape(view)
        source = torch.tensor([shape[axis] for shape in source_shapes], device=device).reshape(batch, *([1] * ndim))
        target = torch.tensor([shape[axis] for shape in target_shapes], device=device).reshape(batch, *([1] * ndim))
        coordinate = position * (source - 1) / (target - 1).clamp_min(1)
        coordinate = torch.minimum(coordinate, source - 1)
        vectors.append((2 * (coordinate + 0.5) / canvas[axis] - 1).expand(batch, *output))
    return torch.stack(vectors[::-1], dim=-1)


def _batch_low_resolution(images: torch.Tensor, plans: list[SpatialImagePlan]) -> torch.Tensor:
    indices = [index for index, plan in enumerate(plans) if plan.lowres_shape is not None]
    if not indices:
        return images
    shapes = [plans[index].lowres_shape for index in indices]
    assert all(shape is not None for shape in shapes)
    sizes = [tuple(shape) for shape in shapes if shape is not None]
    full = tuple(images.shape[2:])
    small = tuple(max(shape[axis] for shape in sizes) for axis in range(len(full)))
    selection = torch.tensor(indices, device=images.device)
    selected = images if len(indices) == len(plans) else images.index_select(0, selection)
    down = _resize_grid([full] * len(sizes), sizes, full, small, images.device)
    up = _resize_grid(sizes, [full] * len(sizes), small, full, images.device)
    reduced = F.grid_sample(selected, down, mode="bilinear", padding_mode="border", align_corners=False)
    restored = F.grid_sample(reduced, up, mode="bilinear", padding_mode="border", align_corners=False)
    return restored if len(indices) == len(plans) else images.index_copy(0, selection, restored)


@torch.no_grad()
def apply_spatial_image_plans(images: torch.Tensor, plans: list[SpatialImagePlan]) -> torch.Tensor:
    """Execute each spatial stage once per batch, with independent sample parameters."""
    if images.ndim not in {4, 5} or images.shape[0] != len(plans):
        raise ValueError("Spatial replay requires one plan per BCHW/BCDHW image")
    images = _orient_batch(images, plans)
    images = _batch_warp(images, plans, "affine_grid")
    images = _batch_low_resolution(images, plans)
    images = _batch_warp(images, plans, "elastic_grid")
    if any(plan.blur_sigmas is not None for plan in plans):
        images = _batch_gaussian_blur(images, [plan.blur_sigmas or (0.0,) * (images.ndim - 2) for plan in plans])
    return images


@torch.no_grad()
def apply_spatial_image_plan(images: torch.Tensor, plan: SpatialImagePlan) -> torch.Tensor:
    """Transform one BCHW/BCDHW image without reading device scalars on the host."""
    if images.ndim not in {4, 5} or images.shape[0] != 1:
        raise ValueError("Spatial replay requires a single BCHW or BCDHW image")
    return apply_spatial_image_plans(_resize_image_group(images, plan.resize_shape), [plan])
