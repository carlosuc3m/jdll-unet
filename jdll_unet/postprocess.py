"""Task-specific postprocessing defaults."""

from __future__ import annotations

import numpy as np

from .errors import InferenceError

try:  # pragma: no cover
    from scipy import ndimage as ndi
except Exception:  # pragma: no cover
    ndi = None


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
    threshold: float = 0.5,
    min_object_size: int = 0,
) -> dict[str, np.ndarray]:
    _validate_threshold(threshold)
    _validate_min_size(min_object_size)
    if foreground_probability.shape != boundary_probability.shape or foreground_probability.ndim not in {2, 3}:
        raise InferenceError("Instance foreground and boundary probabilities must be matching 2D or 3D arrays")
    foreground = foreground_probability >= threshold
    separators = boundary_probability >= threshold
    separated = foreground & ~separators
    separated = _remove_small(separated, min_object_size)
    labels = _label(separated)
    return {
        "foreground_mask": foreground.astype(np.uint8),
        "boundary_mask": separators.astype(np.uint8),
        "labels": labels,
    }
