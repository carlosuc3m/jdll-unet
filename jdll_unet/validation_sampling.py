"""Fixed, bounded volumetric validation plans and crop-first sample loading."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .augment import instance_crop_shape
from .config import ValidationConfig, write_json
from .crop_reading import ResizedCropArray
from .dataset import JdllSegmentationDataset
from .errors import DatasetError, TrainingError
from .geometry import padding_extents
from .io import normalize_image
from .planning import resolve_context_stride
from .targets import prepare_target


@dataclass(frozen=True)
class ValidationDomain:
    item: int
    pair: int
    base_shape: tuple[int, ...]
    shape: tuple[int, ...]
    grid_counts: tuple[int, ...]
    grid_steps: tuple[int, ...]
    grid_offsets: tuple[int, ...]

    @property
    def capacity(self) -> int:
        return math.prod(self.grid_counts)

    def origin(self, cell: int, patch: tuple[int, ...]) -> tuple[int, ...]:
        indices = np.unravel_index(cell, self.grid_counts)
        return tuple(
            int(i * step + offset)
            for i, step, offset in zip(indices, self.grid_steps, self.grid_offsets, strict=True)
        )


class PlannedValidationDataset(Dataset):
    """Keep coordinates, not decoded patches; training samplers remain untouched."""

    def __init__(self, base: JdllSegmentationDataset, options: ValidationConfig, batch_size: int,
                 *, check_cancel: Callable[[], None], emit: Callable[..., Any],
                 path: Path, resume_path: Path | None = None) -> None:
        self.base = base
        self.options = options
        self.dimensions = base.dimensions
        self.patch = tuple(base.augmentation.patch_size)
        self.domains: list[ValidationDomain] = []
        self.path = path
        self.samples: list[tuple[int, int]] = []
        self.forced_indices: list[int] = []
        self.summary: dict[str, Any] = {}
        requested = max(options.minimum_batches, math.ceil(options.minimum_samples / batch_size)) * batch_size
        for item, (pair, _z) in enumerate(base.items):
            check_cancel()
            mask = base.validation_mask(item)
            base_shape = tuple(int(n) for n in mask.shape)
            shape = base_shape
            if base.augmentation.instance_scale_enabled:
                size = base.instance_sizes.get(base.pairs[pair].stem, base.fallback_instance_size)
                if size is None:
                    raise DatasetError("Validation instance scale requires an object size")
                crop = instance_crop_shape(base.augmentation, base_shape, size)
                shape = tuple(max(1, round(n * p / c)) for n, p, c in zip(base_shape, self.patch, crop, strict=True))
            padding_extents(shape, self.patch, base.augmentation.max_padding_ratio)
            counts = tuple(1 + max(0, n - p) // p for n, p in zip(shape, self.patch, strict=True))
            self.domains.append(ValidationDomain(item, pair, base_shape, shape, counts, self.patch, (0,) * len(shape)))
            del mask
        if not self.domains:
            raise DatasetError("No eligible validation targets remain")
        self.sampling_overlap = options.max_sampling_overlap if sum(d.capacity for d in self.domains) < requested else 0.0
        geometry_rng = np.random.default_rng(base.seed)
        for i, domain in enumerate(self.domains):
            steps = tuple(max(1, math.ceil(p * (1 - self.sampling_overlap))) for p in self.patch)
            counts = tuple(1 + max(0, n - p) // step for n, p, step in zip(domain.shape, self.patch, steps, strict=True))
            offsets = tuple(
                -(max(0, p - n) // 2) if n < p else int(geometry_rng.integers(n - ((c - 1) * step + p) + 1))
                for n, p, c, step in zip(domain.shape, self.patch, counts, steps, strict=True)
            )
            self.domains[i] = replace(domain, grid_counts=counts, grid_steps=steps, grid_offsets=offsets)
        self.offsets = np.cumsum([0, *(d.capacity for d in self.domains)], dtype=np.int64)
        self.sources = self._source_records()
        signature = self._signature(batch_size)
        if resume_path is not None:
            if not resume_path.is_file():
                raise TrainingError("Cannot resume: missing fixed validation plan (legacy runs require a new run or fine-tuning)")
            saved = json.loads(resume_path.read_text())
            if saved.get("signature") != signature:
                raise TrainingError("Cannot resume: validation sources, geometry, or sampling policy changed")
            self.samples = [(int(sample[0]), int(sample[1])) for sample in saved["samples"]]
            self.forced_indices = [int(index) for index in saved["forced_sample_indices"]]
            self.summary = saved["summary"]
            if (not self.samples or len(set(self.samples)) != len(self.samples)
                    or len(self.samples) != self.summary["samples"]
                    or len(self.forced_indices) != self.summary["forced_samples"]
                    or len(set(self.forced_indices)) != len(self.forced_indices)
                    or any(not 0 <= i < len(self.samples) for i in self.forced_indices)
                    or any(not 0 <= d < len(self.domains) or not 0 <= cell < self.domains[d].capacity for d, cell in self.samples)):
                raise TrainingError("Cannot resume: invalid saved validation coordinates")
        else:
            self._plan(batch_size, emit, check_cancel)
        write_json(path, {"version": 1, "signature": signature, "sources": self.sources,
                          "patch_size": self.patch, "dimensions": self.dimensions,
                          "domains": [{**asdict(d), "center_z_in_domain": base.items[d.item][1]} for d in self.domains],
                          "samples": self.samples, "forced_sample_indices": self.forced_indices, "summary": self.summary})
        emit("validation_plan", message=f"Fixed validation plan: {len(self)} patches in {math.ceil(len(self) / batch_size)} batches.",
             plan_path=str(path), **self.summary)

    def _source_records(self) -> list[dict[str, Any]]:
        sources = []
        for pair in self.base.pairs:
            files = {}
            for name, path in (("image", pair.image), ("mask", pair.mask)):
                stat = path.stat()
                files[name] = {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
            sources.append({"files": files, "source_id": pair.source_id, "stem": pair.stem,
                            "region": pair.region, "spatial_shape": pair.spatial_shape,
                            "image_axes": pair.image_axes, "mask_axes": pair.mask_axes,
                            "eligible_centers": pair.eligible_centers, "split_origin": pair.split_origin,
                            "resolved_spacing": self.base.case_spacings.get(pair.stem)})
        return sources

    def _signature(self, batch_size: int) -> str:
        policy = {name: getattr(self.options, name) for name in (
            "minimum_batches", "minimum_samples", "foreground_fraction", "minimum_foreground",
            "minimum_source_fraction", "max_sampling_overlap", "candidate_attempts")}
        value = {"version": 1, "sources": self.sources, "domains": [asdict(d) for d in self.domains],
                 "patch": self.patch, "seed": self.base.seed, "policy": policy, "batch": batch_size,
                 "task": self.base.task, "labels": self.base.label_values,
                 "context": [self.base.context_slices, self.base.context_stride_policy,
                             self.base.context_stride, self.base.context_target_spacing],
                 "spacing": self.base.target_spacing, "case_spacings": self.base.case_spacings}
        value["scale_policy"] = {key: getattr(self.base.augmentation, key) for key in (
            "instance_scale_enabled", "target_object_diameter_px", "min_effective_scale", "max_effective_scale", "max_padding_ratio")}
        preparation = self.base.reader.session.annotations
        value["repair_disconnected_instances"] = preparation.config.repair_disconnected_instances if preparation else None
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

    def _cell(self, flat: int) -> tuple[int, int]:
        domain = int(np.searchsorted(self.offsets, flat, side="right") - 1)
        return domain, int(flat - self.offsets[domain])

    def _random_cells(self, count: int, rng: np.random.Generator, excluded: set[int]) -> list[int]:
        # Floyd sampling needs O(count) memory even for a huge spatial lattice.
        total = int(self.offsets[-1])
        count = min(count, total - len(excluded))
        available = total - len(excluded)
        ranks: set[int] = set()
        for j in range(available - count, available):
            draw = int(rng.integers(j + 1))
            ranks.add(j if draw in ranks else draw)
        blocked = sorted(excluded)
        values = []
        for rank in sorted(ranks):
            value = rank
            for missing in blocked:
                if missing > value:
                    break
                value += 1
            values.append(value)
        return [values[i] for i in rng.permutation(len(values))]

    def _plan(self, batch_size: int, emit: Callable[..., Any], check_cancel: Callable[[], None]) -> None:
        cfg = self.options
        requested = max(cfg.minimum_batches, math.ceil(cfg.minimum_samples / batch_size)) * batch_size
        capacity = int(self.offsets[-1])
        count = min(requested, capacity)
        rng = np.random.default_rng(self.base.seed)
        small = capacity < requested
        quota = 0 if small else math.ceil(cfg.foreground_fraction * count)
        by_pair: dict[int, list[int]] = defaultdict(list)
        for i, domain in enumerate(self.domains):
            pair = self.base.pairs[domain.pair]
            z = self.base.items[domain.item][1]
            counts = pair.plane_positive_counts
            positive_count = counts[z] if counts and z is not None else sum(counts)
            if not counts or positive_count > 0:
                by_pair[domain.pair].append(i)
        candidates: dict[int, tuple[float, int]] = {}
        attempts = 0
        pair_order = [int(i) for i in rng.permutation(list(by_pair))]
        requested_coverage = math.ceil(cfg.minimum_source_fraction * len(by_pair)) if quota else 0
        coverage = min(quota, requested_coverage)
        qualifying = 0
        qualifying_sources: set[int] = set()
        # Round-robin source attempts avoid spending the whole quota on one volume.
        for attempt in range(quota * cfg.candidate_attempts):
            check_cancel()
            if not pair_order:
                break
            pair_index = pair_order[attempt % len(pair_order)]
            domain_index = int(rng.choice(by_pair[pair_index]))
            domain = self.domains[domain_index]
            mask = self.base.validation_mask(domain.item)
            sampler = self.base._sampler(domain.item, mask)
            if not sampler.has_foreground:
                continue
            if len(sampler.coordinates):
                point = sampler.coordinates[int(rng.integers(len(sampler.coordinates)))]
            else:
                box = sampler.boxes[int(rng.integers(len(sampler.boxes)))]
                point = (box[:, 0] + box[:, 1] - 1) / 2
            point = (point + 0.5) * np.array(domain.shape) / domain.base_shape - 0.5
            grid = []
            for x, p, cells, step, offset in zip(point, self.patch, domain.grid_counts, domain.grid_steps, domain.grid_offsets, strict=True):
                if cells == 1:
                    grid.append(0)
                else:
                    x -= offset
                    low = max(0, math.ceil((x - p + 1) / step))
                    high = min(cells - 1, math.floor(x / step))
                    grid.append(int(rng.integers(low, high + 1)) if low <= high else int(np.clip(round(x / step), 0, cells - 1)))
            cell = int(np.ravel_multi_index(tuple(grid), domain.grid_counts))
            flat = int(self.offsets[domain_index]) + cell
            if flat in candidates:
                continue
            attempts += 1
            _, raw, valid = self.read_tile(domain_index, domain.origin(cell, self.patch), image=False)
            occupancy = np.count_nonzero(raw[valid]) / max(1, np.count_nonzero(valid))
            candidates[flat] = (float(occupancy), pair_index)
            del mask, raw, valid
            if occupancy >= cfg.minimum_foreground:
                qualifying += 1
                qualifying_sources.add(pair_index)
                if qualifying >= quota and len(qualifying_sources) >= coverage:
                    break
        positive = sorted((flat for flat, (fraction, _) in candidates.items() if fraction > 0),
                          key=lambda flat: (-candidates[flat][0], flat))
        # Retain the strongest candidate from enough distinct sources first, then
        # fill by occupancy. The weakest selected positive defines the fallback.
        selected: list[int] = []
        represented: set[int] = set()
        for flat in positive:
            pair_index = candidates[flat][1]
            if pair_index not in represented and len(represented) < coverage:
                selected.append(flat)
                represented.add(pair_index)
        selected_set = set(selected)
        selected.extend(flat for flat in positive if flat not in selected_set)
        selected = selected[:quota]
        threshold = min([cfg.minimum_foreground, *(candidates[flat][0] for flat in selected)]) if selected else None
        forced_cells = set(selected)
        selected.extend(self._random_cells(count - len(selected), rng, set(selected)))
        order = rng.permutation(len(selected))
        self.samples = [self._cell(selected[i]) for i in order]
        self.forced_indices = [index for index, i in enumerate(order) if selected[i] in forced_cells]
        forced = min(quota, len(positive))
        forced_sources = len({candidates[flat][1] for flat in selected[:forced]})
        self.summary = {
            "scope": "regular_patches", "batch_size": batch_size, "requested_samples": requested,
            "samples": len(self.samples), "batches": math.ceil(len(self.samples) / batch_size),
            "distinct_grid_capacity": capacity, "limited_capacity": small,
            "requested_forced_samples": math.ceil(cfg.foreground_fraction * count),
            "forced_samples": forced, "requested_minimum_foreground": cfg.minimum_foreground,
            "resolved_minimum_foreground": threshold, "foreground_requirements_removed": small,
            "eligible_foreground_sources": len(by_pair), "requested_forced_sources": requested_coverage,
            "feasible_forced_source_target": coverage,
            "forced_sources": forced_sources, "sources_used": len({self.domains[d].pair for d, _ in self.samples}),
            "maximum_sampling_overlap_per_axis": cfg.max_sampling_overlap,
            "resolved_sampling_overlap_per_axis": self.sampling_overlap,
            "candidate_masks_checked": attempts, "candidate_attempt_budget": quota * cfg.candidate_attempts,
            "coordinate_system": "nominal_model_grid", "sampling": "fixed_random_spatial_lattice",
            "jitter": False,
        }
        if small or forced < quota or forced_sources < requested_coverage or (threshold is not None and threshold < cfg.minimum_foreground):
            emit("warning", message="Validation sampling used the bounded capacity/foreground fallback; see validation_plan.json.",
                 **self.summary)

    def __len__(self) -> int:
        return len(self.samples)

    def read_tile(self, domain_index: int, origin: tuple[int, ...], *, image: bool = True,
                  shape: tuple[int, ...] | None = None, labels: bool = True) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
        domain = self.domains[domain_index]
        extent = shape or self.patch
        selection = tuple(slice(max(0, start), min(n, start + size))
                          for start, size, n in zip(origin, extent, domain.shape, strict=True))
        pads = tuple((max(0, -start), max(0, start + size - n))
                     for start, size, n in zip(origin, extent, domain.shape, strict=True))
        if any(s.start >= s.stop for s in selection):
            raise DatasetError("Validation crop has no real target support")
        if labels:
            source_mask = self.base.validation_mask(domain.item)
            raw = ResizedCropArray(source_mask, domain.shape, mask=True)[selection]
        else:
            raw = np.zeros(tuple(s.stop - s.start for s in selection), dtype=np.uint8)
        valid = np.pad(np.ones(raw.shape, dtype=bool), pads)
        raw = np.pad(raw, pads) if any(a or b for a, b in pads) else np.ascontiguousarray(raw)
        if not image:
            return None, raw, valid
        _pair, source_image, _mask, statistics, _ids = self.base._load_item(domain.item)
        tile = ResizedCropArray(source_image, domain.shape)[(slice(None), *selection)]
        if statistics is not None:
            tile = normalize_image(tile, statistics=statistics)
        if any(a or b for a, b in pads):
            tile = np.pad(tile, ((0, 0), *pads))
        return np.ascontiguousarray(tile, dtype=np.float32), raw, valid

    def target(self, mask: np.ndarray, valid: np.ndarray) -> dict[str, torch.Tensor]:
        target = prepare_target(self.base.task, mask, label_values=self.base.label_values,
                                spacing=self.base.target_spacing if self.dimensions == "3d" else None,
                                validity=valid, canonicalize_instances=False)
        assert isinstance(target, dict)
        return {key: torch.from_numpy(value) for key, value in target.items()}

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        domain_index, cell = self.samples[index]
        origin = self.domains[domain_index].origin(cell, self.patch)
        image, mask, valid = self.read_tile(domain_index, origin)
        assert image is not None
        if not mask.flags.writeable:
            mask = mask.copy()
        return torch.from_numpy(image), self.target(mask, valid), torch.from_numpy(mask)

    def provenance(self, domain_index: int, origin: tuple[int, ...], shape: tuple[int, ...]) -> dict[str, Any]:
        domain = self.domains[domain_index]
        base = self.base
        pair = base.pairs[domain.pair]
        native = pair.domain_shape[-len(self.patch):] or domain.base_shape
        spacing = base.case_spacings.get(pair.stem, (1.0, 1.0, 1.0))[-len(self.patch):]
        spacing_ratios = [(n - 1) / (b - 1) if b > 1 else 0.0
                          for n, b in zip(native, domain.base_shape, strict=True)]
        instance_ratios = [b / m for b, m in zip(domain.base_shape, domain.shape, strict=True)]
        ratios = [r * s for r, s in zip(spacing_ratios, instance_ratios, strict=True)]
        offsets = [(s - 1) * r / 2 for s, r in zip(instance_ratios, spacing_ratios, strict=True)]
        metadata = {**base.provenance(domain.item), "model_grid_origin": origin, "model_grid_shape": domain.shape,
                "shape": shape, "axes": "ZYX" if self.dimensions == "3d" else "YX",
                "spacing": [s * r if r > 0 else s for s, r in zip(spacing, ratios, strict=True)],
                "source_voxels_per_output_voxel": ratios,
                "source_coordinate_offset": offsets,
                "coordinate_rule": "source_domain_coordinate = model_grid_coordinate * source_voxels_per_output_voxel + source_coordinate_offset; clamp at edges; add heldout region origin",
                "resampling": "spacing_align_endpoints_then_instance_half_pixel",
                "spacing_grid_shape": domain.base_shape,
                "context_slices": base.context_slices if self.dimensions == "2.5d" else None,
                "context_stride_policy": base.context_stride_policy if self.dimensions == "2.5d" else None,
                "context_target_spacing": base.context_target_spacing if self.dimensions == "2.5d" else None}
        if self.dimensions == "2.5d":
            center = base.items[domain.item][1]
            assert center is not None
            stride = resolve_context_stride(base.context_stride_policy, fixed_stride=base.context_stride,
                                            target_spacing=base.context_target_spacing,
                                            z_spacing=base.case_spacings.get(pair.stem, (1, 1, 1))[0])
            radius = base.context_slices // 2
            source_z_offset = pair.region[0][0] if pair.region else 0
            metadata["context_stride"] = stride
            metadata["context_z_indices"] = [z + source_z_offset if 0 <= z < pair.domain_shape[0] else None
                                              for z in range(center - radius * stride, center + radius * stride + 1, stride)]
            metadata["input_channel_order"] = "modality_major_then_context_z"
        return metadata
