"""Exact, bounded-workspace statistics shared by source inspection and sizing."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from .io import validate_mask_labels

STATISTICS_CHUNK_PIXELS = 1024**2
_DENSE_LABEL_LIMIT = 65536


@dataclass(frozen=True, slots=True)
class LabelRegion:
    count: int
    bounds: tuple[tuple[int, int], ...]

    def touches_border(self, shape: tuple[int, ...]) -> bool:
        return any(start == 0 or end == size for (start, end), size in zip(self.bounds, shape, strict=True))


@dataclass(frozen=True, slots=True)
class MaskAnalysis:
    shape: tuple[int, ...]
    objects: dict[int, LabelRegion]
    planes: tuple[dict[int, LabelRegion], ...]
    foreground_indices: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.uint32), compare=False)

    @property
    def labels(self) -> tuple[int, ...]:
        return tuple(sorted(self.objects))

    @property
    def plane_positive_counts(self) -> tuple[int, ...]:
        return tuple(sum(region.count for region in plane.values()) for plane in self.planes)


def _merge(regions: dict[int, LabelRegion], label: int, count: int, bounds: tuple[tuple[int, int], ...]) -> None:
    previous = regions.get(label)
    if previous is not None:
        count += previous.count
        bounds = tuple(
            (min(a, c), max(b, d)) for (a, b), (c, d) in zip(previous.bounds, bounds, strict=True)
        )
    regions[label] = LabelRegion(count, bounds)


def analyze_mask(mask: np.ndarray, path: Path | None = None) -> MaskAnalysis:
    """Accumulate all IDs together, never allocating an array indexed by a large source ID.

    Per-plane summaries also supply the representative cross section for 2.5D.
    Temporary arrays are tile-sized, including for transposed arrays and memmaps.
    """
    if mask.ndim not in (2, 3) or any(size < 1 for size in mask.shape):
        raise ValueError("Mask analysis requires a nonempty Y,X or Z,Y,X array")
    planes = []
    rng = np.random.default_rng(0)
    reservoir = np.empty(0, dtype=np.uint32 if mask.size <= 2**32 else np.uint64)
    seen = 0
    limit = 100_000
    objects: dict[int, LabelRegion] = {}
    height, width = mask.shape[-2:]
    columns = min(width, STATISTICS_CHUNK_PIXELS)
    rows = max(1, STATISTICS_CHUNK_PIXELS // columns)
    for z, plane in enumerate(mask[None] if mask.ndim == 2 else mask):
        regions: dict[int, LabelRegion] = {}
        for y in range(0, height, rows):
            for x in range(0, width, columns):
                tile = plane[y : y + rows, x : x + columns]
                validate_mask_labels(tile, path or Path("<array>"))
                maximum = int(tile.max())
                if maximum == 0:
                    continue
                positions = np.flatnonzero(tile)
                count = len(positions)
                keep = min(limit, seen + count)
                take_old = int(rng.hypergeometric(seen, count, keep)) if seen else 0
                old = reservoir[rng.choice(len(reservoir), take_old, replace=False)] if take_old else reservoir[:0]
                selected = positions[rng.choice(count, keep - take_old, replace=False)]
                flat = (z * height + y + selected // tile.shape[1]) * width + x + selected % tile.shape[1]
                reservoir = np.concatenate((old, flat.astype(reservoir.dtype)))
                seen += count
                if maximum <= _DENSE_LABEL_LIMIT:
                    labels = tile.astype(np.intp, copy=False)
                    counts = np.bincount(labels.reshape(-1), minlength=maximum + 1)
                    boxes = ndi.find_objects(labels, max_label=maximum)
                    for label in np.flatnonzero(counts[1:]) + 1:
                        box = boxes[label - 1]
                        bounds = ((y + box[0].start, y + box[0].stop), (x + box[1].start, x + box[1].stop))
                        _merge(regions, int(label), int(counts[label]), bounds)
                else:
                    values, inverse, counts = np.unique(tile, return_inverse=True, return_counts=True)
                    labels = inverse.reshape(tile.shape) + 1
                    boxes = ndi.find_objects(labels, max_label=len(values))
                    for index, value in enumerate(values):
                        if value == 0:
                            continue
                        box = boxes[index]
                        bounds = ((y + box[0].start, y + box[0].stop), (x + box[1].start, x + box[1].stop))
                        _merge(regions, int(value), int(counts[index]), bounds)
        planes.append(regions)
        if mask.ndim == 3:
            for label, region in regions.items():
                _merge(objects, label, region.count, ((z, z + 1), *region.bounds))
    return MaskAnalysis(tuple(mask.shape), planes[0] if mask.ndim == 2 else objects, tuple(planes), reservoir)
