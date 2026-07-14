"""Informative semantic-region scale diagnostics for training and inference."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import ndimage as ndi


def _region_measures(binary: npt.NDArray[np.bool_]) -> tuple[list[int], list[int]]:
    labels, count = ndi.label(binary, structure=ndi.generate_binary_structure(binary.ndim, 1))
    if count == 0:
        return [], []
    sizes = np.bincount(labels.ravel())[1:]
    border_ids: set[int] = set()
    for axis in range(binary.ndim):
        border_ids.update(int(value) for value in np.unique(np.take(labels, (0, -1), axis=axis)) if value)
    all_sizes = [int(value) for value in sizes]
    interior_sizes = [int(value) for index, value in enumerate(sizes, start=1) if index not in border_ids]
    return interior_sizes, all_sizes


def _summary(primary: list[float], fallback: list[float], border_count: int) -> dict[str, Any]:
    selected = primary or fallback
    values = np.asarray(selected, dtype=np.float64)
    result: dict[str, Any] = {
        "count": len(selected),
        "interior_count": len(primary),
        "all_count": len(fallback),
        "border_touching_count": border_count,
        "border_touching_fraction": border_count / len(fallback) if fallback else 0.0,
        "used_border_fallback": not primary and bool(fallback),
    }
    for name, percentile in (("p10", 10), ("p25", 25), ("median", 50), ("p75", 75), ("p90", 90)):
        result[name] = float(np.percentile(values, percentile)) if values.size else None
    return result


def semantic_scale_diagnostics(
    masks: Iterable[npt.NDArray[Any]],
    *,
    dimensions: str,
    patch_size: tuple[int, ...],
    label_values: Iterable[int],
) -> dict[str, Any]:
    """Summarize connected semantic-region occupancy relative to the model patch."""

    if dimensions not in {"2d", "2.5d", "3d"}:
        raise ValueError(f"Unsupported dimensions: {dimensions}")
    denominator = float(np.prod(patch_size[-2:] if dimensions == "2.5d" else patch_size))
    classes = sorted({int(value) for value in label_values if int(value) != 0})
    keys: list[int | str] = ["foreground", *classes]
    primary: dict[int | str, list[float]] = {key: [] for key in keys}
    fallback: dict[int | str, list[float]] = {key: [] for key in keys}
    border_counts: dict[int | str, int] = dict.fromkeys(keys, 0)
    cases = 0

    for mask in masks:
        cases += 1
        planes = mask if dimensions == "2.5d" else (mask,)
        for plane in planes:
            for key in keys:
                binary = plane > 0 if key == "foreground" else plane == key
                interior, all_regions = _region_measures(binary)
                primary[key].extend(value / denominator for value in interior)
                fallback[key].extend(value / denominator for value in all_regions)
                border_counts[key] += len(all_regions) - len(interior)

    return {
        "measure": "connected_region_fraction",
        "reference_domain": "resampled_model_space",
        "reference": {
            "2d": "patch_xy_area",
            "2.5d": "center_slice_patch_xy_area",
            "3d": "patch_volume",
        }[dimensions],
        "patch_size": list(patch_size),
        "cases_measured": cases,
        "pooled_foreground": _summary(primary["foreground"], fallback["foreground"], border_counts["foreground"]),
        "per_class": {str(key): _summary(primary[key], fallback[key], border_counts[key]) for key in classes},
    }


def compare_semantic_region_fraction(value: float, diagnostics: dict[str, Any]) -> dict[str, Any]:
    """Compare an inference estimate with the pooled training distribution."""

    if not np.isfinite(value) or value <= 0:
        raise ValueError("semantic region fraction must be a positive finite number")
    stats = diagnostics.get("pooled_foreground", {})
    median, p10, p90 = stats.get("median"), stats.get("p10"), stats.get("p90")
    if median is None:
        return {"provided_fraction": value, "status": "training_distribution_unavailable", "warning": False}
    status = "within_training_p10_p90"
    if p10 is not None and value < float(p10):
        status = "below_training_p10"
    elif p90 is not None and value > float(p90):
        status = "above_training_p90"
    return {
        "provided_fraction": value,
        "training_median": float(median),
        "training_p10": float(p10) if p10 is not None else None,
        "training_p90": float(p90) if p90 is not None else None,
        "ratio_to_training_median": value / float(median),
        "status": status,
        "warning": status != "within_training_p10_p90",
    }
