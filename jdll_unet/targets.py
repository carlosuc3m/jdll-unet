"""Mask-to-target conversion for supported segmentation tasks."""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover
    from scipy import ndimage as ndi
except Exception:  # pragma: no cover
    ndi = None


def binary_target(mask: np.ndarray) -> np.ndarray:
    return (mask != 0).astype(np.float32)[None, ...]


def multiclass_target(mask: np.ndarray, label_values: list[int] | None = None) -> np.ndarray:
    if label_values is None:
        labels = sorted(int(v) for v in np.unique(mask) if int(v) != 0)
    else:
        labels = [int(v) for v in label_values if int(v) != 0]
    out = np.zeros(mask.shape, dtype=np.int64)
    for index, label in enumerate(labels, start=1):
        out[mask == label] = index
    return out


def boundary_target(mask: np.ndarray, width: int = 1) -> np.ndarray:
    labels = mask.astype(np.int64, copy=False)
    boundary = np.zeros(labels.shape, dtype=bool)
    for axis in range(labels.ndim):
        before = [slice(None)] * labels.ndim
        after = [slice(None)] * labels.ndim
        before[axis] = slice(0, -1)
        after[axis] = slice(1, None)
        differences = labels[tuple(before)] != labels[tuple(after)]
        boundary[tuple(before)] |= differences
        boundary[tuple(after)] |= differences
    boundary &= labels != 0
    if width > 1 and ndi is not None:
        boundary = ndi.binary_dilation(boundary, iterations=int(width) - 1)
    return boundary.astype(np.float32)[None, ...]


def instance_targets(mask: np.ndarray, boundary_width: int = 1) -> dict[str, np.ndarray]:
    return {
        "foreground": binary_target(mask),
        "boundary": boundary_target(mask, width=boundary_width),
    }


def prepare_target(
    task: str,
    mask: np.ndarray,
    label_values: list[int] | None = None,
    boundary_width: int = 1,
) -> np.ndarray | dict[str, np.ndarray]:
    if task == "binary_semantic":
        return binary_target(mask)
    if task == "multiclass_semantic":
        return multiclass_target(mask, label_values=label_values)
    if task == "instance_friendly":
        return instance_targets(mask, boundary_width=boundary_width)
    raise ValueError(f"Unsupported task: {task}")


def target_output_channels(task: str, label_values: list[int] | None = None) -> int:
    if task == "binary_semantic":
        return 1
    if task == "instance_friendly":
        return 2
    if task == "multiclass_semantic":
        return (len(label_values or []) + 1) if label_values else 2
    raise ValueError(f"Unsupported task: {task}")
