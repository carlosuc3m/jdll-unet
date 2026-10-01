"""Instance-size measurement and scale normalization for 2D, 2.5D and 3D."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi

from .io import validate_mask_labels
from .label_statistics import LabelRegion, MaskAnalysis, analyze_mask


@dataclass(frozen=True, slots=True)
class InstanceSizeEstimate:
    median_diameter_px: float
    sampled_instances: int
    available_instances: int
    median_principal_axes: tuple[float, ...] | None = None


@dataclass(frozen=True, slots=True)
class InstanceLabelRepair:
    labels: np.ndarray
    original_labels: int
    repaired_components: int


def canonicalize_instance_volume(mask: np.ndarray) -> InstanceLabelRepair:
    """Convert binary/disconnected 3D labels into unique connected instance IDs."""

    if mask.ndim != 3:
        raise ValueError("Instance volume canonicalization requires a Z,Y,X mask")
    from .annotations import component_labels_and_sources

    validate_mask_labels(mask, Path("<array>"))
    output, source_ids = component_labels_and_sources(mask)
    source_labels = {int(value) for value in source_ids[1:]}
    next_label = max(source_labels, default=0) + 1
    repaired = 0
    seen = set()
    for component, value in enumerate(source_ids[1:], start=1):
        label = int(value)
        if label in seen:
            if next_label > np.iinfo(np.int64).max:
                raise ValueError("Cannot repair instance labels beyond the int64 range")
            source_ids[component] = next_label
            next_label += 1
            repaired += 1
        seen.add(label)
    flat = output.reshape(-1)
    for start in range(0, flat.size, 1024**2):
        chunk = flat[start : start + 1024**2]
        chunk[:] = source_ids[chunk]
    return InstanceLabelRepair(output, len(source_labels), repaired)


def _principal_axes(
    mask: np.ndarray, label: int, region: LabelRegion, spacing: tuple[float, ...]
) -> tuple[float, ...]:
    crop = mask[tuple(slice(start, end) for start, end in region.bounds)]
    count = 0
    mean = np.zeros(crop.ndim)
    products = np.zeros((crop.ndim, crop.ndim))
    # Merge centered moments from bounded coordinate blocks, never a whole object cloud.
    for start in range(0, crop.size, 1024**2):
        indexes = np.flatnonzero(crop.flat[start : start + 1024**2] == label)
        if not len(indexes):
            continue
        coords = np.asarray(np.unravel_index(indexes + start, crop.shape), dtype=np.float64).T
        coords *= np.asarray(spacing)
        block_mean = coords.mean(axis=0)
        coords -= block_mean
        delta = block_mean - mean
        total = count + len(indexes)
        products += coords.T @ coords + np.outer(delta, delta) * (count * len(indexes) / total)
        mean += delta * (len(indexes) / total)
        count = total
    covariance = products / (count - 1) if count > 1 else products
    return tuple(float(value) for value in 2.0 * np.sqrt(np.maximum(np.linalg.eigvalsh(covariance), 0.0)))


def estimate_from_analysis(
    analysis: MaskAnalysis,
    *,
    mask: np.ndarray | None = None,
    volume_xy: bool = False,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    max_instances: int = 21,
    exclude_border: bool = True,
    min_size: int = 4,
    seed: int = 0,
    measure: str = "equivalent_sphere_diameter",
) -> InstanceSizeEstimate | None:
    """Select eligible identities before doing any object-specific coordinate work."""
    if max_instances < 1:
        raise ValueError("max_instances must be positive")
    if any(not math.isfinite(value) or value <= 0 for value in spacing):
        raise ValueError("spacing must contain finite positive values")
    volume = len(analysis.shape) == 3
    if volume_xy and not volume:
        raise ValueError("Volume cross-section estimation requires a Z,Y,X mask")
    complete: list[tuple[int, LabelRegion, int | None]] = []
    border: list[tuple[int, LabelRegion, int | None]] = []
    for label in analysis.labels:
        region = analysis.objects[label]
        selected_z = None
        if volume_xy:
            first_z, end_z = region.bounds[0]
            touches = first_z == 0 or end_z == analysis.shape[0]
            sections = [(z, plane[label]) for z, plane in enumerate(analysis.planes) if label in plane]
            if touches:
                selected_z, region = max(sections, key=lambda item: item[1].count)
            else:
                center = (first_z + end_z - 1) / 2.0
                selected_z, region = min(sections, key=lambda item: (abs(item[0] - center), -item[1].count))
            if exclude_border and region.touches_border(analysis.shape[-2:]):
                continue
        else:
            touches = region.touches_border(analysis.shape)
            if not volume and exclude_border and touches:
                continue
            touches = touches and exclude_border
        if region.count >= min_size:
            (border if touches else complete).append((label, region, selected_z))
    selected: list[tuple[int, LabelRegion, int | None]] = []
    rng = np.random.default_rng(seed)
    for candidates in (complete, border):
        remaining = max_instances - len(selected)
        if remaining <= 0:
            break
        indexes = np.arange(len(candidates)) if len(candidates) <= remaining else rng.choice(len(candidates), remaining, replace=False)
        selected.extend(candidates[int(index)] for index in indexes)
    if not selected:
        return None
    values = []
    axes_values = []
    for label, region, selected_z in selected:
        if measure == "principal_axes":
            if mask is None:
                raise ValueError("Principal-axis estimation requires the source mask")
            axes = _principal_axes(
                mask if selected_z is None else mask[selected_z], label, region,
                spacing if volume and not volume_xy else (1.0, 1.0),
            )
            axes_values.append(axes)
            values.append(float(np.median(axes)))
        elif volume and not volume_xy:
            values.append((6.0 * region.count * math.prod(spacing) / math.pi) ** (1.0 / 3.0))
        else:
            values.append(math.sqrt(4.0 * region.count / math.pi))
    median_axes = (
        tuple(float(value) for value in np.median(np.asarray(axes_values), axis=0))
        if axes_values and volume and not volume_xy else None
    )
    return InstanceSizeEstimate(float(np.median(values)), len(selected), len(complete) + len(border), median_axes)


def estimate_volume_instance_size(
    mask: np.ndarray,
    max_instances: int = 21,
    exclude_xy_border: bool = True,
    min_instance_area: int = 4,
    seed: int = 0,
    measure: str = "equivalent_sphere_diameter",
    *,
    canonicalize_instances: bool = True,
) -> tuple[InstanceSizeEstimate | None, InstanceLabelRepair]:
    """Estimate one XY object diameter for a complete instance volume."""

    if mask.ndim != 3:
        raise ValueError("Volume instance size estimation requires a Z,Y,X mask")
    repair = canonicalize_instance_volume(mask) if canonicalize_instances else None
    labels = repair.labels if repair is not None else mask
    analysis = analyze_mask(labels)
    return estimate_from_analysis(
        analysis, mask=labels, volume_xy=True, max_instances=max_instances,
        exclude_border=exclude_xy_border, min_size=min_instance_area, seed=seed, measure=measure,
    ), repair or InstanceLabelRepair(mask, len(analysis.objects), 0)


def estimate_instance_size(
    mask: np.ndarray,
    max_instances: int = 21,
    exclude_border: bool = True,
    min_instance_area: int = 4,
    seed: int = 0,
    measure: str = "equivalent_sphere_diameter",
    *,
    canonicalize_instances: bool = True,
) -> InstanceSizeEstimate | None:
    """Estimate median equivalent diameter from a reproducible instance sample."""

    if mask.ndim != 2:
        raise ValueError("Instance size estimation currently supports 2D masks only")
    analysis = analyze_mask(mask)
    components = mask
    if canonicalize_instances and analysis.labels == (1,):
        components = ndi.label(mask != 0)[0]
        analysis = analyze_mask(components)
    return estimate_from_analysis(
        analysis, mask=components, max_instances=max_instances,
        exclude_border=exclude_border, min_size=min_instance_area, seed=seed, measure=measure,
    )


def estimate_3d_instance_size(
    mask: np.ndarray,
    spacing: tuple[float, float, float],
    max_instances: int = 21,
    exclude_border: bool = True,
    min_instance_voxels: int = 4,
    seed: int = 0,
    measure: str = "equivalent_sphere_diameter",
    *,
    canonicalize_instances: bool = True,
) -> tuple[InstanceSizeEstimate | None, InstanceLabelRepair]:
    """Estimate physical 3D instance size, preferring complete objects."""

    if mask.ndim != 3:
        raise ValueError("3D instance size estimation requires a Z,Y,X mask")
    repair = canonicalize_instance_volume(mask) if canonicalize_instances else None
    labels = repair.labels if repair is not None else mask
    analysis = analyze_mask(labels)
    return estimate_from_analysis(
        analysis, mask=labels, spacing=spacing, max_instances=max_instances,
        exclude_border=exclude_border, min_size=min_instance_voxels, seed=seed, measure=measure,
    ), repair or InstanceLabelRepair(mask, len(analysis.objects), 0)


def resize_3d_pair_to_shape(
    image: np.ndarray, mask: np.ndarray, target: tuple[int, int, int]
) -> tuple[np.ndarray, np.ndarray]:
    if target == mask.shape:
        return image, mask
    image_t = torch.from_numpy(np.ascontiguousarray(image[None].astype(np.float32, copy=False)))
    mask_t = torch.from_numpy(np.ascontiguousarray(mask[None, None].astype(np.float32, copy=False)))
    resized_image = F.interpolate(image_t, size=target, mode="trilinear", align_corners=False)[0].numpy()
    resized_mask = F.interpolate(mask_t, size=target, mode="nearest")[0, 0].numpy().astype(mask.dtype, copy=False)
    return np.ascontiguousarray(resized_image), np.ascontiguousarray(resized_mask)


def resize_2d_pair(image: np.ndarray, mask: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Resize an image/mask pair while preserving image values and integer labels."""

    if scale <= 0 or not np.isfinite(scale):
        raise ValueError("scale must be a finite positive number")
    target = (max(1, int(round(mask.shape[0] * scale))), max(1, int(round(mask.shape[1] * scale))))
    return resize_2d_pair_to_shape(image, mask, target)


def resize_2d_pair_to_shape(
    image: np.ndarray, mask: np.ndarray, target: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Resize an image/mask pair to an exact 2D shape."""

    if target == mask.shape:
        return image, mask
    image_t = torch.from_numpy(np.ascontiguousarray(image[None].astype(np.float32, copy=False)))
    mask_t = torch.from_numpy(np.ascontiguousarray(mask[None, None].astype(np.float32, copy=False)))
    resized_image = F.interpolate(image_t, size=target, mode="bilinear", align_corners=False)[0].numpy()
    resized_mask = F.interpolate(mask_t, size=target, mode="nearest")[0, 0].numpy().astype(mask.dtype, copy=False)
    return np.ascontiguousarray(resized_image), np.ascontiguousarray(resized_mask)


def resize_2d_channels(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize C,Y,X floating-point channels with bilinear interpolation."""

    tensor = torch.from_numpy(np.ascontiguousarray(array[None].astype(np.float32, copy=False)))
    return F.interpolate(tensor, size=shape, mode="bilinear", align_corners=False)[0].numpy()


def aggregate_instance_statistics(estimates: list[InstanceSizeEstimate]) -> dict[str, object]:
    diameters = np.asarray([item.median_diameter_px for item in estimates], dtype=np.float64)
    sampled = np.asarray([item.sampled_instances for item in estimates], dtype=np.int64)
    if not estimates:
        return {"images_measured": 0}
    return {
        "images_measured": len(estimates),
        "median_object_diameter_px": float(np.median(diameters)),
        "object_diameter_p10_px": float(np.percentile(diameters, 10)),
        "object_diameter_p90_px": float(np.percentile(diameters, 90)),
        "median_instances_sampled_per_image": float(np.median(sampled)),
        "minimum_instances_sampled_per_image": int(sampled.min()),
        "maximum_instances_sampled_per_image": int(sampled.max()),
    }


def estimate_to_dict(estimate: InstanceSizeEstimate) -> dict[str, object]:
    return asdict(estimate)
