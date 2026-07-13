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


def resize_2d_pair(image: np.ndarray, mask: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Resize an image/mask pair while preserving image values and integer labels."""

    if scale <= 0 or not np.isfinite(scale):
        raise ValueError("scale must be a finite positive number")
    target = tuple(max(1, int(round(size * scale))) for size in mask.shape)
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
