"""Dataset-driven segmentation task detection."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median

import numpy as np

from .config import architecture_defaults
from .errors import TaskDetectionError
from .io import ImageMaskPair, discover_dataset, load_mask, read_class_names

try:  # pragma: no cover - fallback exists for minimal environments
    from scipy import ndimage as ndi
except Exception:  # pragma: no cover
    ndi = None


SUPPORTED_TASKS = {"binary_semantic", "multiclass_semantic", "instance_friendly"}


@dataclass(slots=True)
class MaskStats:
    path: str
    unique_nonzero_labels: list[int]
    connected_components_per_label: dict[int, int]
    labels_are_sequential_ids: bool
    many_components_share_one_label_value: bool


def _component_count(binary: np.ndarray) -> int:
    if not np.any(binary):
        return 0
    if ndi is not None:
        _, count = ndi.label(binary)
        return int(count)
    seen = np.zeros(binary.shape, dtype=bool)
    count = 0
    offsets: list[tuple[int, ...]] = []
    for axis in range(binary.ndim):
        for delta in (-1, 1):
            offset_vector = [0] * binary.ndim
            offset_vector[axis] = delta
            offsets.append(tuple(offset_vector))
    for start in np.ndindex(binary.shape):
        if not binary[start] or seen[start]:
            continue
        count += 1
        stack: list[tuple[int, ...]] = [start]
        seen[start] = True
        while stack:
            current = stack.pop()
            for neighbor_offset in offsets:
                neighbor = tuple(coord + step for coord, step in zip(current, neighbor_offset, strict=True))
                if all(0 <= coord < limit for coord, limit in zip(neighbor, binary.shape, strict=True)) and binary[neighbor] and not seen[neighbor]:
                    seen[neighbor] = True
                    stack.append(neighbor)
    return count


def mask_statistics(mask: np.ndarray, path: str = "") -> MaskStats:
    labels = sorted(int(v) for v in np.unique(mask) if int(v) != 0)
    components = {label: _component_count(mask == label) for label in labels}
    sequential = bool(labels) and labels == list(range(1, len(labels) + 1))
    many_components = any(count > 3 for count in components.values())
    return MaskStats(
        path=path,
        unique_nonzero_labels=labels,
        connected_components_per_label=components,
        labels_are_sequential_ids=sequential,
        many_components_share_one_label_value=many_components,
    )


def _metadata_signals(dataset_path: Path) -> dict[str, object]:
    lower_names = {path.name.lower() for path in dataset_path.rglob("*") if path.is_file()}
    class_names = read_class_names(dataset_path)
    roi_signal = any("roi" in name and name.endswith((".zip", ".roi", ".json")) for name in lower_names)
    boxes_signal = any("box" in name or "bbox" in name or "bounding" in name for name in lower_names)
    points_signal = any("point" in name or "centroid" in name for name in lower_names)
    return {
        "class_names": class_names,
        "annotation_source": "roi_manager_one_roi_per_object" if roi_signal else None,
        "bounding_boxes": boxes_signal,
        "points": points_signal,
    }


def _all_label_sets(stats: Iterable[MaskStats]) -> list[set[int]]:
    return [set(item.unique_nonzero_labels) for item in stats]


def detect_task_from_pairs(
    pairs: list[ImageMaskPair],
    dataset_path: Path | str | None = None,
    requested_task: str = "auto",
    dimensions: str | None = None,
) -> dict[str, object]:
    """Infer the task from mask statistics and lightweight metadata."""

    requested_task = {"classes": "multiclass_semantic", "objects": "instance_friendly"}.get(
        requested_task,
        requested_task,
    )
    if requested_task in SUPPORTED_TASKS:
        return {
            "task": requested_task,
            "ambiguous": False,
            "reason": "Task was supplied explicitly.",
            "score": None,
        }
    if not pairs:
        raise TaskDetectionError("Cannot detect task without image/mask pairs")

    dataset_root = Path(dataset_path) if dataset_path is not None else pairs[0].image.parent.parent
    metadata = _metadata_signals(dataset_root)
    if metadata["bounding_boxes"]:
        return {
            "task": "unsupported",
            "route": "yolo",
            "ambiguous": False,
            "reason": "Bounding-box annotations are better handled by a detection backend.",
        }
    if metadata["points"]:
        return {
            "task": "unsupported",
            "route": "detection",
            "ambiguous": False,
            "reason": "Point annotations require a detection workflow rather than this UNet backend.",
        }

    stats = [mask_statistics(load_mask(pair.mask, dimensions=dimensions), str(pair.mask)) for pair in pairs]
    label_sets = _all_label_sets(stats)
    all_labels = sorted(set().union(*label_sets)) if label_sets else []
    non_empty_label_sets = [labels for labels in label_sets if labels]
    median_unique = median([len(labels) for labels in label_sets]) if label_sets else 0
    consistent = len({tuple(sorted(labels)) for labels in non_empty_label_sets}) <= 1
    small_stable = bool(non_empty_label_sets) and consistent and len(set().union(*non_empty_label_sets)) <= 8
    one_component_values = 0
    total_values = 0
    for item in stats:
        for count in item.connected_components_per_label.values():
            total_values += 1
            if count <= 1:
                one_component_values += 1
    most_one_component = total_values > 0 and one_component_values / total_values >= 0.7
    sequential_ids = sum(item.labels_are_sequential_ids and len(item.unique_nonzero_labels) > 3 for item in stats)
    many_components_same_label = sum(item.many_components_share_one_label_value for item in stats)

    score = 0
    reasons: list[str] = []
    if metadata["annotation_source"] == "roi_manager_one_roi_per_object":
        score += 4
        reasons.append("ROI-manager style object annotations detected.")
    if median_unique > 10:
        score += 3
        reasons.append("Masks contain many unique labels per image.")
    if most_one_component and median_unique > 3:
        score += 3
        reasons.append("Most label values have one connected component.")
    if not consistent and non_empty_label_sets:
        score += 2
        reasons.append("Label values are not stable across images.")
    if sequential_ids:
        score += 2
        reasons.append("Masks look like sequential object-id label images.")
    if metadata["class_names"]:
        score -= 4
        reasons.append("Class names metadata exists.")
    if small_stable:
        score -= 3
        reasons.append("The label set is small and stable across images.")
    if many_components_same_label:
        score -= 2
        reasons.append("Many components share the same label value.")

    if not all_labels or all_labels == [1]:
        task = "binary_semantic"
        ambiguous = False
    elif score >= 4:
        task = "instance_friendly"
        ambiguous = False
    elif score <= -2:
        task = "multiclass_semantic"
        ambiguous = False
    else:
        task = "ambiguous"
        ambiguous = True

    result: dict[str, object] = {
        "task": task,
        "ambiguous": ambiguous,
        "score": score,
        "reason": " ".join(reasons) or "Masks use foreground/background labels.",
        "stats": [asdict(item) for item in stats],
        "median_unique_labels_per_image": float(median_unique),
        "unique_label_values": all_labels,
        "class_names": metadata["class_names"],
    }
    if ambiguous:
        result.update(
            {
                "question": "Do different numbers in the annotation represent different biological classes or different individual objects?",
                "choices": ["Different classes.", "Different objects."],
                "class_choice_task": "multiclass_semantic",
                "object_choice_task": "instance_friendly",
            }
        )
    return result


def detect_task(config: dict | str | Path) -> dict[str, object]:
    if isinstance(config, (str, Path)):
        dataset_path = Path(config)
        requested = "auto"
    else:
        if "dataset_path" not in config:
            raise TaskDetectionError("detect_task config requires dataset_path")
        dataset_path = Path(config["dataset_path"])
        requested = str(config.get("task", "auto"))
        if requested == "classes":
            requested = "multiclass_semantic"
        elif requested == "objects":
            requested = "instance_friendly"
    splits = discover_dataset(dataset_path)
    dimensions = (
        architecture_defaults(str(config["architecture"])).dimensions
        if isinstance(config, dict) and config.get("architecture") is not None
        else None
    )
    return detect_task_from_pairs(
        splits.train + splits.val,
        dataset_path,
        requested_task=requested,
        dimensions=dimensions,
    )
