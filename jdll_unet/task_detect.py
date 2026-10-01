"""Dataset-driven segmentation task detection."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median

import numpy as np

from .annotations import AnnotationPreparation, component_labels_and_sources
from .config import architecture_defaults
from .errors import TaskDetectionError
from .geometry import load_domain_mask
from .io import ImageMaskPair, discover_dataset, read_class_names, validate_mask_labels

SUPPORTED_TASKS = {"binary_semantic", "multiclass_semantic", "instance_friendly"}


@dataclass(slots=True)
class MaskStats:
    path: str
    unique_nonzero_labels: list[int]
    connected_components_per_label: dict[int, int]
    labels_are_sequential_ids: bool
    many_components_share_one_label_value: bool
    connectivity_analyzed: bool = True


def _mask_stats(labels: list[int], path: str, components: dict[int, int] | None = None) -> MaskStats:
    sequential = bool(labels) and labels == list(range(1, len(labels) + 1))
    return MaskStats(
        path=path,
        unique_nonzero_labels=labels,
        connected_components_per_label=components if components is not None else {},
        labels_are_sequential_ids=sequential,
        many_components_share_one_label_value=any(count > 3 for count in (components or {}).values()),
        connectivity_analyzed=components is not None,
    )


def mask_statistics(mask: np.ndarray, path: str = "", *, analyze_connectivity: bool = True) -> MaskStats:
    validate_mask_labels(mask, Path(path))
    if analyze_connectivity:
        _, source_ids = component_labels_and_sources(mask)
        components = dict(Counter(int(value) for value in source_ids[1:]))
        return _mask_stats(sorted(components), path, components)
    return _mask_stats([int(value) for value in np.unique(mask) if value != 0], path)


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
    preparation: AnnotationPreparation | None = None,
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

    if preparation is None:
        stats = [
            mask_statistics(
                load_domain_mask(pair, dimensions=dimensions, original=True, raw=True),
                str(pair.mask), analyze_connectivity=False,
            )
            for pair in pairs
        ]
    else:
        stats = []
        for pair in pairs:
            components = preparation.cached_detection_components(pair, dimensions)
            stats.append(_mask_stats(
                list(preparation.detection_labels(pair, dimensions)), str(pair.mask), components,
            ))
    label_sets = _all_label_sets(stats)
    all_labels = sorted(set().union(*label_sets)) if label_sets else []
    non_empty_label_sets = [labels for labels in label_sets if labels]
    median_unique = median([len(labels) for labels in label_sets]) if label_sets else 0
    consistent = len({tuple(sorted(labels)) for labels in non_empty_label_sets}) <= 1
    small_stable = bool(non_empty_label_sets) and consistent and len(set().union(*non_empty_label_sets)) <= 8
    score = 0
    reasons: list[str] = []
    if metadata["annotation_source"] == "roi_manager_one_roi_per_object":
        score += 4
        reasons.append("ROI-manager style object annotations detected.")
    if median_unique > 10:
        score += 3
        reasons.append("Masks contain many unique labels per image.")
    if not consistent and non_empty_label_sets:
        score += 2
        reasons.append("Label values are not stable across images.")
    if metadata["class_names"]:
        score -= 4
        reasons.append("Class names metadata exists.")
    if small_stable:
        score -= 3
        reasons.append("The label set is small and stable across images.")
    # Connectivity contributes at most +3/-2. Skip it only when those bounds
    # cannot change the task, preserving the existing classification thresholds.
    binary = not all_labels or all_labels == [1]
    decided = binary or score - 2 >= 4 or score + (3 if median_unique > 3 else 0) <= -2
    if not decided:
        for index, (pair, item) in enumerate(zip(pairs, stats, strict=True)):
            if item.connectivity_analyzed:
                continue
            if preparation is not None:
                components = preparation.detection_components(pair, dimensions)
                stats[index] = _mask_stats(item.unique_nonzero_labels, item.path, components)
            else:
                stats[index] = mask_statistics(
                    load_domain_mask(pair, dimensions=dimensions, original=True, raw=True), str(pair.mask)
                )
    if all(item.connectivity_analyzed for item in stats):
        component_counts = [count for item in stats for count in item.connected_components_per_label.values()]
        if component_counts and median_unique > 3 and sum(count <= 1 for count in component_counts) / len(component_counts) >= 0.7:
            score += 3
            reasons.append("Most label values have one connected component.")
        if any(item.many_components_share_one_label_value for item in stats):
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
