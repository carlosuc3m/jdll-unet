"""Regular single-forward validation and optional crop-first full diagnostics."""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .annotations import _storage_dtype
from .config import PostprocessingConfig
from .crop_reading import CropArray
from .device_ops import autocast_context
from .errors import TrainingError
from .geometry import load_domain_mask, padding_extents
from .infer import _tile_layout
from .losses import primary_logits
from .metrics import compute_metrics, primary_metric
from .planning import restore_continuous_maps
from .targets import binary_target, boundary_target, normalized_instance_distance
from .validation_metrics import ValidationAccumulator
from .validation_previews import PreviewAnchor, available_ram, importance_map, predict_tile, source_crop_workspace
from .validation_sampling import PlannedValidationDataset


def regular_validation(model: torch.nn.Module, loader: DataLoader, data: PlannedValidationDataset,
                       device: torch.device, epoch: int, *, weights: dict[str, float], focal_gamma: float,
                       focal_alpha: float | None, preview_count: int, progress_interval: int,
                       emit: Callable[..., Any], check_cancel: Callable[[], None],
                       dtype: torch.dtype = torch.float32) -> tuple[dict, dict, list[PreviewAnchor]]:
    model.eval()
    accumulator = ValidationAccumulator(data.base.task, weights, focal_gamma, focal_alpha)
    indices = set(np.linspace(0, len(data) - 1, min(preview_count, len(data)), dtype=int).tolist())
    anchors: list[PreviewAnchor] = []
    retained = 0
    emit("validation", status="started", epoch=epoch, current=0, maximum=len(data), unit="patches",
         message=f"Starting regular validation for epoch {epoch}: {len(data)} patches.")
    with torch.inference_mode():
        for step, (images_cpu, targets_cpu, labels_cpu) in enumerate(loader, start=1):
            check_cancel()
            images = images_cpu.to(device, non_blocking=True)
            targets = {key: value.to(device, non_blocking=True) for key, value in targets_cpu.items()}
            with autocast_context(device, dtype):
                logits = model(images)
            first = accumulator.samples
            accumulator.update(logits, targets, cpu_validity=targets_cpu.get("valid"))
            main = primary_logits(logits)
            for offset in range(images.shape[0]):
                sample = first + offset
                if sample not in indices:
                    continue
                needed = (images_cpu[offset].numel() * 4 + labels_cpu[offset].numel() * labels_cpu.element_size()
                          + targets_cpu["valid"][offset].numel() + main[offset].numel() * 4)
                available = available_ram()
                if retained + needed > int(data.options.preview_max_bytes) or (available is not None and needed > available // 4):
                    emit("warning", epoch=epoch, message="Reduced preview anchor retention to respect available memory.")
                    continue
                anchor = PreviewAnchor(sample, images_cpu[offset].numpy().copy(), labels_cpu[offset].numpy().copy(),
                                       targets_cpu["valid"][offset, 0].numpy().copy(),
                                       main[offset].detach().float().cpu().numpy().copy())
                anchors.append(anchor)
                retained += anchor.nbytes
            del logits, main, images, targets, images_cpu, targets_cpu, labels_cpu
            if step % progress_interval == 0 or accumulator.samples == len(data):
                emit("validation", status="progress", epoch=epoch, current=accumulator.samples, maximum=len(data),
                     unit="patches", batches=step, message=f"Validated {accumulator.samples}/{len(data)} patches.")
                check_cancel()
    losses, metrics = accumulator.result()
    emit("validation", status="completed", epoch=epoch, current=accumulator.samples, maximum=len(data), unit="patches",
         batches=accumulator.batches, losses=losses, metrics=metrics, aggregation=accumulator.aggregation(),
         message=f"Regular validation completed for epoch {epoch}.")
    return losses, metrics, anchors


def _native_mask(data: PlannedValidationDataset, domain_index: int) -> np.ndarray:
    base = data.base
    domain = data.domains[domain_index]
    pair = base.pairs[domain.pair]
    center = base.items[domain.item][1]
    if pair.mask_axes is not None:
        source = CropArray(pair, base.reader, mask=True, original_mask=base.task != "instance_friendly", center_z=center)
        return source[(slice(None),) * len(domain.shape)]
    mask = load_domain_mask(pair, base.dimensions, base.reader, raw=True, original=base.task != "instance_friendly")
    return mask[center] if center is not None else mask


def _full_target(data: PlannedValidationDataset, domain_index: int, mask: np.ndarray,
                 spacing: tuple[float, ...]) -> dict[str, torch.Tensor]:
    base = data.base
    if base.task == "binary_semantic":
        return {"semantic": torch.from_numpy(binary_target(mask)[None])}
    if base.task == "multiclass_semantic":
        labels = [value for value in (base.label_values or []) if value != 0]
        semantic = np.zeros(mask.shape, dtype=_storage_dtype(len(labels)))
        for index, label in enumerate(labels, start=1):
            semantic[mask == label] = index
        return {"semantic": torch.from_numpy(semantic[None])}
    domain = data.domains[domain_index]
    center = base.items[domain.item][1]
    analysis = base.mask_analysis(domain.pair, mask if center is None else None)
    objects = analysis.planes[center] if center is not None and len(analysis.shape) == 3 else analysis.objects
    regions = ((label, tuple(slice(a, b) for a, b in region.bounds)) for label, region in objects.items())
    # Preserve compact source labels and cached bounds; no int64 relabeling or
    # array indexed by the largest (possibly sparse) annotation ID.
    target = {"foreground": binary_target(mask), "boundary": boundary_target(mask),
              "distance": normalized_instance_distance(mask, spacing=spacing, regions=regions),
              "instances": (mask if mask.flags.writeable else mask.copy())[None]}
    return {key: torch.from_numpy(value[None]) for key, value in target.items()}


def full_validation(model: torch.nn.Module, data: PlannedValidationDataset, device: torch.device, epoch: int,
                    postprocessing: PostprocessingConfig, *, emit: Callable[..., Any],
                    check_cancel: Callable[[], None], dtype: torch.dtype = torch.float32) -> dict[str, Any]:
    """One held-out domain at a time; only tiles and accumulators, no full image."""
    model.eval()
    importance = importance_map(data.patch, data.options.tile_blending)
    per_case: dict[str, list[dict[str, float]]] = {}
    evaluated: dict[str, list[int] | None] = {}
    for index, domain in enumerate(data.domains):
        check_cancel()
        pair = data.base.pairs[domain.pair]
        native_shape = pair.domain_shape[-len(data.patch):] or domain.base_shape
        output_channels = 3 if data.base.task == "instance_friendly" else len(data.base.label_values or [1]) + 1 if data.base.task == "multiclass_semantic" else 1
        architecture = getattr(model, "config", None)
        if architecture is not None:
            output_channels = architecture.output_channels
        layout = _tile_layout(domain.shape, data.patch, data.options.tile_overlap)
        pads = padding_extents(domain.shape, data.patch, data.base.augmentation.max_padding_ratio)
        # Whole-domain instance reconstruction still has sizeable native work
        # arrays. Reject unsafe diagnostics before allocating them, not after OOM.
        working = math.prod(layout.padded_shape) * (4 * output_channels + 4)
        working += math.prod(native_shape) * (128 if data.base.task == "instance_friendly" else 24 + 12 * output_channels)
        working += source_crop_workspace(data, index)
        available = available_ram()
        if available is not None and working > available * 0.6:
            raise MemoryError(f"Full validation of {pair.stem} needs an estimated {working / 1024**2:.0f} MiB; "
                              f"only {available / 1024**2:.0f} MiB host RAM is available. Regular checkpoints are preserved.")
        accum = np.zeros((output_channels, *layout.padded_shape), dtype=np.float32)
        counts = np.zeros(layout.padded_shape, dtype=np.float32)
        total_tiles = math.prod(len(starts) for starts in layout.starts_by_axis)
        for tile_index, padded_origin in enumerate(itertools.product(*layout.starts_by_axis), start=1):
            check_cancel()
            origin = tuple(start - pad[0] for start, pad in zip(padded_origin, pads, strict=True))
            image, _mask, _valid = data.read_tile(index, origin, labels=False)
            assert image is not None
            prediction = predict_tile(model, image, device, dtype)
            selection = tuple(slice(start, start + p) for start, p in zip(padded_origin, data.patch, strict=True))
            accum[(slice(None), *selection)] += prediction * importance
            counts[selection] += importance
            del image, _mask, _valid, prediction
            if tile_index % 10 == 0 or tile_index == total_tiles:
                emit("full_validation", status="progress", epoch=epoch, current=index, maximum=len(data.domains),
                     case=pair.stem, completed_tiles=tile_index, total_tiles=total_tiles, unit="domains",
                     message=f"Full validation {pair.stem}: {tile_index}/{total_tiles} tiles.")
        accum /= counts[None]
        del counts
        selection = tuple(slice(pad[0], pad[0] + n) for pad, n in zip(pads, domain.shape, strict=True))
        logits = accum[(slice(None), *selection)]
        if tuple(logits.shape[1:]) != tuple(native_shape):
            logits = restore_continuous_maps(logits, tuple(native_shape))
        check_cancel()
        mask = _native_mask(data, index)
        spacing = data.base.case_spacings.get(pair.stem, (1.0, 1.0, 1.0))[-mask.ndim:]
        target = _full_target(data, index, mask, spacing)
        metrics = compute_metrics(data.base.task, torch.from_numpy(logits[None]), target,
                                  postprocessing={**asdict(postprocessing), "spacing": spacing})
        if not all(math.isfinite(value) for value in metrics.values()):
            raise TrainingError(f"Non-finite full-validation metrics for {pair.stem}")
        per_case.setdefault(pair.stem, []).append(metrics)
        z = data.base.provenance(domain.item)["original_z_index"]
        if z is None:
            evaluated[pair.stem] = None
        else:
            centers = evaluated.setdefault(pair.stem, [])
            assert centers is not None
            centers.append(z)
        del accum, logits, mask, target
        emit("full_validation", status="progress", epoch=epoch, current=index + 1, maximum=len(data.domains),
             case=pair.stem, metrics=metrics, unit="domains", message=f"Full validation completed domain {index + 1}/{len(data.domains)}.")
    means = {name: {key: float(np.mean([item[key] for item in values])) for key in values[0]}
             for name, values in per_case.items()}
    scores = {name: primary_metric(data.base.task, values) for name, values in means.items()}
    skipped_centers = {}
    if data.dimensions == "2.5d":
        for pair in data.base.pairs:
            offset = pair.region[0][0] if pair.region else 0
            completed = set(evaluated.get(pair.stem) or [])
            skipped_centers[pair.stem] = [z + offset for z in range(pair.domain_shape[0]) if z + offset not in completed]
    return {"mean_dice": float(np.mean(list(scores.values()))), "per_case_dice": scores,
            "per_case_metrics": means, "evaluated_centers": evaluated, "evaluated_domains": len(data.domains),
            "skipped_domains": 0, "scope": "full_heldout_domains", "aggregation": "mean_per_case; mean_eligible_centers_within_case",
            "skipped_centers": skipped_centers, "skipped_center_reason": "not_eligible_in_resolved_validation_geometry",
            "postprocessing": asdict(postprocessing)}
