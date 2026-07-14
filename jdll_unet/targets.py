"""Mask-to-target conversion for supported segmentation tasks."""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover
    from scipy import ndimage as ndi
except Exception:  # pragma: no cover
    ndi = None


def binary_target(mask: np.ndarray) -> np.ndarray:
    return (mask != 0).astype(np.float32)[None, ...]


def canonical_instance_labels(mask: np.ndarray) -> np.ndarray:
    """Give every face-connected component a unique ID, including binary annotations."""

    if ndi is None:
        return mask.astype(np.int64, copy=False)
    output = np.zeros(mask.shape, dtype=np.int64)
    next_id = 1
    structure = ndi.generate_binary_structure(mask.ndim, 1)
    for source_id in (int(value) for value in np.unique(mask) if int(value) != 0):
        components, count = ndi.label(mask == source_id, structure=structure)
        for component in range(1, int(count) + 1):
            output[components == component] = next_id
            next_id += 1
    return output


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
    """Mixed boundary: outside ring, two-sided ID interfaces, and object voxels at array edges."""

    labels = mask.astype(np.int64, copy=False)
    boundary = np.zeros(labels.shape, dtype=bool)
    for axis in range(labels.ndim):
        before: list[slice | int] = [slice(None)] * labels.ndim
        after: list[slice | int] = [slice(None)] * labels.ndim
        before[axis] = slice(0, -1)
        after[axis] = slice(1, None)
        left = labels[tuple(before)]
        right = labels[tuple(after)]
        differences = left != right
        both_objects = differences & (left != 0) & (right != 0)
        left_object = differences & (left != 0) & (right == 0)
        right_object = differences & (left == 0) & (right != 0)
        # Touching IDs are marked on both object sides; outer contours use the background side.
        boundary[tuple(before)] |= both_objects | right_object
        boundary[tuple(after)] |= both_objects | left_object
        first: list[slice | int] = [slice(None)] * labels.ndim
        last: list[slice | int] = [slice(None)] * labels.ndim
        first[axis] = 0
        last[axis] = -1
        boundary[tuple(first)] |= labels[tuple(first)] != 0
        boundary[tuple(last)] |= labels[tuple(last)] != 0
    if width > 1 and ndi is not None:
        boundary = ndi.binary_dilation(boundary, iterations=int(width) - 1)
    return boundary.astype(np.float32)[None, ...]


def normalized_instance_distance(
    mask: np.ndarray,
    spacing: tuple[float, ...] | None = None,
) -> np.ndarray:
    target = np.zeros(mask.shape, dtype=np.float32)
    if ndi is None:
        return target[None, ...]
    for instance_id in np.unique(mask):
        if int(instance_id) == 0:
            continue
        instance = mask == instance_id
        distance = ndi.distance_transform_edt(instance, sampling=spacing).astype(np.float32)
        maximum = float(distance.max())
        if maximum > 0:
            target[instance] = distance[instance] / maximum
    return target[None, ...]


def instance_targets(
    mask: np.ndarray,
    boundary_width: int = 1,
    spacing: tuple[float, ...] | None = None,
) -> dict[str, np.ndarray]:
    mask = canonical_instance_labels(mask)
    return {
        "foreground": binary_target(mask),
        "boundary": boundary_target(mask, width=boundary_width),
        "distance": normalized_instance_distance(mask, spacing=spacing),
        "instances": mask.astype(np.int64, copy=False)[None, ...],
    }


def prepare_target(
    task: str,
    mask: np.ndarray,
    label_values: list[int] | None = None,
    boundary_width: int = 1,
    spacing: tuple[float, ...] | None = None,
) -> np.ndarray | dict[str, np.ndarray]:
    if task == "binary_semantic":
        return binary_target(mask)
    if task == "multiclass_semantic":
        return multiclass_target(mask, label_values=label_values)
    if task == "instance_friendly":
        return instance_targets(mask, boundary_width=boundary_width, spacing=spacing)
    raise ValueError(f"Unsupported task: {task}")


def target_output_channels(task: str, label_values: list[int] | None = None) -> int:
    if task == "binary_semantic":
        return 1
    if task == "instance_friendly":
        return 3
    if task == "multiclass_semantic":
        return (len(label_values or []) + 1) if label_values else 2
    raise ValueError(f"Unsupported task: {task}")
