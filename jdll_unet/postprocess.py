"""Task-specific postprocessing defaults."""

from __future__ import annotations

from typing import Any

import numpy as np

from .errors import InferenceError

try:  # pragma: no cover
    from scipy import ndimage as ndi
except Exception:  # pragma: no cover
    ndi = None

try:  # pragma: no cover
    from skimage.morphology import h_maxima as _h_maxima
    from skimage.segmentation import watershed as _watershed
    h_maxima: Any = _h_maxima
    watershed: Any = _watershed
except Exception:  # pragma: no cover
    h_maxima = watershed = None


def _validate_threshold(threshold: float) -> None:
    if not 0 <= threshold <= 1:
        raise InferenceError("threshold must be in [0, 1]")


def _validate_min_size(min_object_size: int) -> None:
    if min_object_size < 0:
        raise InferenceError("min_object_size cannot be negative")


def _remove_small(binary: np.ndarray, min_size: int) -> np.ndarray:
    if min_size <= 0 or ndi is None:
        return binary
    labels, count = ndi.label(binary)
    if count == 0:
        return binary
    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_size
    keep[0] = False
    return keep[labels]


def _label(binary: np.ndarray) -> np.ndarray:
    if ndi is None:
        return binary.astype(np.uint16)
    labels, _ = ndi.label(binary)
    return labels.astype(np.uint32, copy=False)


def postprocess_binary(
    probability: np.ndarray,
    threshold: float = 0.5,
    min_object_size: int = 0,
    fill_holes: bool = False,
    connected_components: bool = True,
) -> dict[str, np.ndarray]:
    _validate_threshold(threshold)
    _validate_min_size(min_object_size)
    if probability.ndim not in {2, 3}:
        raise InferenceError(f"Binary probability map must be 2D or 3D, got shape {probability.shape}")
    binary = probability >= threshold
    if fill_holes and ndi is not None:
        binary = ndi.binary_fill_holes(binary)
    binary = _remove_small(binary, min_object_size)
    result = {"mask": binary.astype(np.uint8)}
    if connected_components:
        result["labels"] = _label(binary)
    return result


def postprocess_multiclass(probabilities: np.ndarray, min_object_size: int = 0) -> dict[str, np.ndarray]:
    _validate_min_size(min_object_size)
    if probabilities.ndim not in {3, 4}:
        raise InferenceError(f"Multiclass probabilities must be C,Y,X or C,Z,Y,X, got shape {probabilities.shape}")
    labels = np.argmax(probabilities, axis=0).astype(np.uint16)
    if min_object_size > 0 and ndi is not None:
        cleaned = np.zeros_like(labels)
        for cls in range(1, int(labels.max()) + 1):
            cleaned[_remove_small(labels == cls, min_object_size)] = cls
        labels = cleaned
    return {"mask": labels}


def postprocess_instance(
    foreground_probability: np.ndarray,
    boundary_probability: np.ndarray,
    distance_probability: np.ndarray | None = None,
    threshold: float = 0.5,
    min_object_size: int = 0,
    method: str = "distance_boundary_watershed",
    seed_distance_threshold: float = 0.35,
    seed_boundary_threshold: float = 0.5,
    seed_h: float = 0.1,
    min_seed_size: int = 3,
    boundary_weight: float = 1.0,
    connectivity: str = "face",
    spacing: tuple[float, ...] | None = None,
    min_object_size_physical: float | None = None,
    min_seed_size_physical: float | None = None,
    **_unused: object,
) -> dict[str, np.ndarray]:
    _validate_threshold(threshold)
    _validate_min_size(min_object_size)
    if foreground_probability.shape != boundary_probability.shape or foreground_probability.ndim not in {2, 3}:
        raise InferenceError("Instance foreground and boundary probabilities must be matching 2D or 3D arrays")
    voxel_measure = float(np.prod(spacing)) if spacing is not None else 1.0
    if min_object_size_physical is not None:
        min_object_size = max(min_object_size, int(np.ceil(min_object_size_physical / voxel_measure)))
    if min_seed_size_physical is not None:
        min_seed_size = max(min_seed_size, int(np.ceil(min_seed_size_physical / voxel_measure)))
    foreground = foreground_probability >= threshold
    separators = boundary_probability >= threshold
    foreground = _remove_small(foreground, min_object_size)
    if method == "connected_components" or distance_probability is None:
        labels = _label(foreground & ~separators)
    else:
        if distance_probability.shape != foreground_probability.shape:
            raise InferenceError("Instance distance probability must match foreground shape")
        if h_maxima is None or watershed is None or ndi is None:
            raise InferenceError("distance_boundary_watershed requires scipy and scikit-image")
        clean = np.clip(distance_probability, 0, 1) * np.clip(foreground_probability, 0, 1) * (1 - np.clip(boundary_probability, 0, 1))
        clean[~foreground] = 0
        seeds = h_maxima(clean, h=seed_h)
        seeds &= clean >= seed_distance_threshold
        seeds &= boundary_probability < seed_boundary_threshold
        seeds &= foreground
        seeds = _remove_small(seeds, min_seed_size)
        rank = foreground.ndim
        structure = ndi.generate_binary_structure(rank, 1 if connectivity == "face" else rank)
        markers, _ = ndi.label(seeds, structure=structure)
        components, count = ndi.label(foreground, structure=structure)
        next_marker = int(markers.max()) + 1
        for component_id in range(1, count + 1):
            component = components == component_id
            if np.any(markers[component]):
                continue
            values = np.where(component, clean, -np.inf)
            index = int(np.argmax(values))
            if np.isfinite(values.flat[index]):
                markers.flat[index] = next_marker
                next_marker += 1
        labels = watershed(-clean + boundary_weight * boundary_probability, markers, mask=foreground, connectivity=structure)
        cleaned = np.zeros(labels.shape, dtype=np.uint32)
        next_id = 1
        for instance_id in range(1, int(labels.max()) + 1):
            region = labels == instance_id
            if int(region.sum()) >= min_object_size:
                cleaned[region] = next_id
                next_id += 1
        labels = cleaned
    return {
        "foreground_mask": foreground.astype(np.uint8),
        "boundary_mask": separators.astype(np.uint8),
        "labels": labels,
    }
