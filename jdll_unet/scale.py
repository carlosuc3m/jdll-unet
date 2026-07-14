"""2D instance-size measurement and scale normalization utilities."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi


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
    source_labels = [int(value) for value in np.unique(mask) if int(value) != 0]
    output = np.zeros(mask.shape, dtype=np.int64)
    next_label = max(source_labels, default=0) + 1
    repaired = 0
    structure = ndi.generate_binary_structure(3, 1)
    for source_label in source_labels:
        components, count = ndi.label(mask == source_label, structure=structure)
        for component in range(1, int(count) + 1):
            assigned = source_label if component == 1 else next_label
            if component > 1:
                next_label += 1
                repaired += 1
            output[components == component] = assigned
    return InstanceLabelRepair(output, len(source_labels), repaired)


def estimate_volume_instance_size(
    mask: np.ndarray,
    max_instances: int = 21,
    exclude_xy_border: bool = True,
    min_instance_area: int = 4,
    seed: int = 0,
    measure: str = "equivalent_sphere_diameter",
) -> tuple[InstanceSizeEstimate | None, InstanceLabelRepair]:
    """Estimate one XY object diameter for a complete instance volume."""

    repair = canonicalize_instance_volume(mask)
    z_size, height, width = mask.shape
    interior: list[float] = []
    z_border: list[float] = []
    for label in (int(value) for value in np.unique(repair.labels) if int(value) != 0):
        object_mask = repair.labels == label
        zz = np.nonzero(object_mask)[0]
        touches_z = zz.min() == 0 or zz.max() == z_size - 1
        if touches_z:
            areas = [(int(z), int(np.count_nonzero(object_mask[z]))) for z in np.unique(zz)]
            selected_z = max(areas, key=lambda item: item[1])[0]
        else:
            center = (int(zz.min()) + int(zz.max())) / 2.0
            candidates = sorted(
                np.unique(zz), key=lambda z: (abs(float(z) - center), -np.count_nonzero(object_mask[z]))
            )
            selected_z = int(candidates[0])
        plane_y, plane_x = np.nonzero(object_mask[selected_z])
        area = len(plane_y)
        if area < min_instance_area:
            continue
        if exclude_xy_border and (
            plane_y.min() == 0
            or plane_x.min() == 0
            or plane_y.max() == height - 1
            or plane_x.max() == width - 1
        ):
            continue
        if measure == "principal_axes":
            coords = np.column_stack((plane_y, plane_x)).astype(np.float64)
            axes = 2.0 * np.sqrt(np.maximum(np.linalg.eigvalsh(np.cov(coords.T)), 0.0)) if len(coords) > 1 else np.zeros(2)
            diameter = float(np.median(axes))
        else:
            diameter = math.sqrt(4.0 * area / math.pi)
        (z_border if touches_z else interior).append(diameter)
    rng = np.random.default_rng(seed)
    selected: list[float] = []
    for candidates in (interior, z_border):
        remaining = max_instances - len(selected)
        if remaining <= 0:
            break
        if len(candidates) <= remaining:
            selected.extend(candidates)
        else:
            indices = rng.choice(len(candidates), size=remaining, replace=False)
            selected.extend(candidates[int(index)] for index in indices)
    available = len(interior) + len(z_border)
    estimate = None
    if selected:
        estimate = InstanceSizeEstimate(float(np.median(selected)), len(selected), available)
    return estimate, repair


def _instance_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    labels = [int(value) for value in np.unique(mask) if int(value) != 0]
    if labels == [1]:
        components, count = ndi.label(mask != 0)
        return components, int(count)
    return mask.astype(np.int64, copy=False), len(labels)


def estimate_instance_size(
    mask: np.ndarray,
    max_instances: int = 21,
    exclude_border: bool = True,
    min_instance_area: int = 4,
    seed: int = 0,
    measure: str = "equivalent_sphere_diameter",
) -> InstanceSizeEstimate | None:
    """Estimate median equivalent diameter from a reproducible instance sample."""

    if mask.ndim != 2:
        raise ValueError("Instance size estimation currently supports 2D masks only")
    components, _ = _instance_components(mask)
    measurements: list[float] = []
    height, width = mask.shape
    for label in (int(value) for value in np.unique(components) if int(value) != 0):
        yy, xx = np.nonzero(components == label)
        area = len(yy)
        if area < min_instance_area:
            continue
        if exclude_border and (yy.min() == 0 or xx.min() == 0 or yy.max() == height - 1 or xx.max() == width - 1):
            continue
        if measure == "principal_axes":
            coords = np.column_stack((yy, xx)).astype(np.float64)
            axes = 2.0 * np.sqrt(np.maximum(np.linalg.eigvalsh(np.cov(coords.T)), 0.0)) if len(coords) > 1 else np.zeros(2)
            measurements.append(float(np.median(axes)))
        else:
            measurements.append(math.sqrt(4.0 * area / math.pi))
    available = len(measurements)
    if available == 0:
        return None
    if available > max_instances:
        rng = np.random.default_rng(seed)
        indices = rng.choice(available, size=max_instances, replace=False)
        measurements = [measurements[int(index)] for index in indices]
    return InstanceSizeEstimate(
        median_diameter_px=float(np.median(measurements)),
        sampled_instances=len(measurements),
        available_instances=available,
    )


def estimate_3d_instance_size(
    mask: np.ndarray,
    spacing: tuple[float, float, float],
    max_instances: int = 21,
    exclude_border: bool = True,
    min_instance_voxels: int = 4,
    seed: int = 0,
    measure: str = "equivalent_sphere_diameter",
) -> tuple[InstanceSizeEstimate | None, InstanceLabelRepair]:
    """Estimate physical 3D instance size, preferring complete objects."""

    repair = canonicalize_instance_volume(mask)
    complete: list[tuple[float, tuple[float, ...]]] = []
    border: list[tuple[float, tuple[float, ...]]] = []
    shape = np.asarray(mask.shape)
    voxel_volume = float(np.prod(spacing))
    for label in (int(value) for value in np.unique(repair.labels) if int(value) != 0):
        coords = np.argwhere(repair.labels == label)
        if len(coords) < min_instance_voxels:
            continue
        touches = bool(np.any(coords.min(axis=0) == 0) or np.any(coords.max(axis=0) == shape - 1))
        physical = coords.astype(np.float64) * np.asarray(spacing)
        axes = tuple(float(value) for value in (2.0 * np.sqrt(np.maximum(np.linalg.eigvalsh(np.cov(physical.T)), 0.0))))
        volume = len(coords) * voxel_volume
        diameter = 2.0 * (3.0 * volume / (4.0 * math.pi)) ** (1.0 / 3.0)
        (border if touches else complete).append((diameter, axes))
    selected: list[tuple[float, tuple[float, ...]]] = []
    rng = np.random.default_rng(seed)
    for candidates in (complete, border):
        remaining = max_instances - len(selected)
        if remaining <= 0:
            break
        indexes = np.arange(len(candidates)) if len(candidates) <= remaining else rng.choice(len(candidates), remaining, replace=False)
        selected.extend(candidates[int(index)] for index in indexes)
    if not selected and border:
        selected = border[:max_instances]
    if not selected:
        return None, repair
    values = [float(np.median(axes)) if measure == "principal_axes" else diameter for diameter, axes in selected]
    median_axes = tuple(float(value) for value in np.median(np.asarray([axes for _diameter, axes in selected]), axis=0))
    return InstanceSizeEstimate(float(np.median(values)), len(selected), len(complete) + len(border), median_axes), repair


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
