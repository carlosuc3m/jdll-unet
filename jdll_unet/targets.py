"""Mask-to-target conversion for supported segmentation tasks."""

from __future__ import annotations

import numpy as np
import torch

from .errors import DatasetError

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


def has_trusted_instance_ids(mask: np.ndarray) -> bool:
    """Check source connectivity once, independently of the numerical label IDs.

    Every nonzero ID must form exactly one face-connected component. A binary
    mask with disconnected objects still requires component canonicalization.
    """
    if ndi is None:
        return False
    compact = compact_instance_labels(mask)
    boxes = ndi.find_objects(compact)
    if not boxes:
        return False
    structure = ndi.generate_binary_structure(mask.ndim, 1)
    return all(
        int(ndi.label(compact[box] == label, structure=structure)[1]) == 1
        for label, box in enumerate(boxes, start=1)
        if box is not None
    )


def compact_instance_labels(mask: np.ndarray) -> np.ndarray:
    """Compact nonzero instance IDs without changing instance membership."""
    labels = np.unique(mask)
    labels = labels[labels != 0]
    if labels.size == 0:
        return np.zeros(mask.shape, dtype=np.int64)
    positions = np.searchsorted(labels, mask)
    valid = mask != 0
    output = np.zeros(mask.shape, dtype=np.int64)
    output[valid] = positions[valid] + 1
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


def boundary_target(mask: np.ndarray, width: int = 1, validity: np.ndarray | None = None) -> np.ndarray:
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
        if validity is not None:
            differences &= validity[tuple(before)] & validity[tuple(after)]
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
        boundary[tuple(first)] |= (labels[tuple(first)] != 0) & (
            validity[tuple(first)] if validity is not None else True
        )
        boundary[tuple(last)] |= (labels[tuple(last)] != 0) & (validity[tuple(last)] if validity is not None else True)
    if width > 1 and ndi is not None:
        boundary = ndi.binary_dilation(boundary, iterations=int(width) - 1)
    if validity is not None:
        boundary &= validity
    return boundary.astype(np.float32)[None, ...]


def normalized_instance_distance(
    mask: np.ndarray,
    spacing: tuple[float, ...] | None = None,
    validity: np.ndarray | None = None,
) -> np.ndarray:
    target = np.zeros(mask.shape, dtype=np.float32)
    if ndi is None:
        return target[None, ...]

    for instance_id, bbox in enumerate(ndi.find_objects(mask), start=1):
        if bbox is None:
            continue
        expanded = tuple(
            slice(max(0, axis.start - 1), min(size, axis.stop + 1)) for axis, size in zip(bbox, mask.shape, strict=True)
        )
        instance = mask[expanded] == instance_id
        distance = ndi.distance_transform_edt(instance, sampling=spacing).astype(np.float32)
        if validity is None:
            maximum = float(distance.max())
        else:
            valid_values = distance[validity[expanded]]
            maximum = float(valid_values.max()) if valid_values.size else 0.0
        if maximum > 0:
            local_target = target[expanded]
            local_target[instance] = distance[instance] / maximum
    return target[None, ...]


def instance_targets(
    mask: np.ndarray,
    boundary_width: int = 1,
    spacing: tuple[float, ...] | None = None,
    validity: np.ndarray | None = None,
    canonicalize_instances: bool = True,
    defer_dense: bool = False,
) -> dict[str, np.ndarray]:
    if defer_dense and boundary_width != 1:
        raise ValueError("Deferred boundary targets require boundary_width=1")
    mask = canonical_instance_labels(mask) if canonicalize_instances else compact_instance_labels(mask)
    distance_mask = mask
    if validity is not None and not np.all(validity) and ndi is not None:
        # Unknown padded support must not introduce an object/background interface.
        nearest = ndi.distance_transform_edt(~validity, return_distances=False, return_indices=True)
        distance_mask = mask[tuple(nearest)]
    result = {
        "distance": normalized_instance_distance(distance_mask, spacing=spacing, validity=validity),
        "instances": mask.astype(np.int64, copy=False)[None, ...],
    }
    if not defer_dense:
        result.update(
            foreground=binary_target(mask), boundary=boundary_target(mask, width=boundary_width, validity=validity)
        )
    return result


def prepare_target(
    task: str,
    mask: np.ndarray,
    label_values: list[int] | None = None,
    boundary_width: int = 1,
    spacing: tuple[float, ...] | None = None,
    validity: np.ndarray | None = None,
    canonicalize_instances: bool = True,
    defer_dense: bool = False,
) -> np.ndarray | dict[str, np.ndarray]:
    if validity is not None:
        if validity.shape != mask.shape or not np.any(validity):
            raise DatasetError("Target requires nonempty real support matching its spatial shape")
        mask = np.where(validity, mask, 0)
        if task == "instance_friendly":
            result = instance_targets(
                mask,
                boundary_width=boundary_width,
                spacing=spacing,
                validity=validity,
                canonicalize_instances=canonicalize_instances,
                defer_dense=defer_dense,
            )
        elif defer_dense:
            result = {"labels": mask.astype(np.int64, copy=False)[None]}
        else:
            semantic = prepare_target(
                task,
                mask,
                label_values=label_values,
                boundary_width=boundary_width,
                spacing=spacing,
            )
            assert isinstance(semantic, np.ndarray)
            result = {"semantic": semantic}
        result["valid"] = np.ascontiguousarray(validity[None], dtype=bool)
        return result
    if task == "binary_semantic":
        return binary_target(mask)
    if task == "multiclass_semantic":
        return multiclass_target(mask, label_values=label_values)
    if task == "instance_friendly":
        return instance_targets(
            mask,
            boundary_width=boundary_width,
            spacing=spacing,
            canonicalize_instances=canonicalize_instances,
        )
    raise ValueError(f"Unsupported task: {task}")


def tensor_boundary_target(labels: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Construct one-voxel instance boundaries for a B1HW/B1DHW batch."""
    boundary = torch.zeros_like(labels, dtype=torch.bool)
    for axis in range(2, labels.ndim):
        before = [slice(None)] * labels.ndim
        after = [slice(None)] * labels.ndim
        before[axis], after[axis] = slice(0, -1), slice(1, None)
        left, right = labels[tuple(before)], labels[tuple(after)]
        different = (left != right) & valid[tuple(before)] & valid[tuple(after)]
        touching = different & (left != 0) & (right != 0)
        boundary[tuple(before)] |= touching | (different & (left == 0) & (right != 0))
        boundary[tuple(after)] |= touching | (different & (left != 0) & (right == 0))
        boundary.select(axis, 0).logical_or_((labels.select(axis, 0) != 0) & valid.select(axis, 0))
        boundary.select(axis, -1).logical_or_((labels.select(axis, -1) != 0) & valid.select(axis, -1))
    return (boundary & valid).float()


@torch.no_grad()
def complete_device_targets(
    task: str, target: dict[str, torch.Tensor], label_values: list[int] | None
) -> dict[str, torch.Tensor]:
    """Build dense targets after transfer without device-to-host scalar reads."""
    if task == "instance_friendly":
        labels, valid = target["instances"], target["valid"].bool()
        return {**target, "foreground": (labels != 0).float(), "boundary": tensor_boundary_target(labels, valid)}
    if "labels" not in target:
        return target
    labels = target["labels"]
    if task == "binary_semantic":
        semantic = (labels != 0).float()
    elif task == "multiclass_semantic":
        semantic = torch.zeros_like(labels[:, 0])
        for index, label in enumerate([value for value in (label_values or []) if value != 0], start=1):
            semantic = torch.where(labels[:, 0] == label, index, semantic)
    else:
        raise ValueError(f"Unsupported task: {task}")
    return {"semantic": semantic, "valid": target["valid"]}


def target_output_channels(task: str, label_values: list[int] | None = None) -> int:
    if task == "binary_semantic":
        return 1
    if task == "instance_friendly":
        return 3
    if task == "multiclass_semantic":
        return (len(label_values or []) + 1) if label_values else 2
    raise ValueError(f"Unsupported task: {task}")
