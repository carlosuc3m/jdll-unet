"""Bounded, reusable crop candidates built from shared source-mask statistics."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .errors import DatasetError
from .geometry import padding_extents, stable_seed
from .label_statistics import MaskAnalysis


@dataclass
class CropSampler:
    analysis: MaskAnalysis
    shape: tuple[int, ...]
    seed: int
    center_z: int | None = None
    instances: bool = False
    pools: OrderedDict = field(default_factory=OrderedDict)

    def __post_init__(self) -> None:
        native_shape = self.analysis.shape
        coordinates = np.column_stack(np.unravel_index(self.analysis.foreground_indices, native_shape))
        regions = self.analysis.objects
        if self.center_z is not None:
            coordinates = coordinates[coordinates[:, 0] == self.center_z, 1:]
            native_shape = native_shape[-2:]
            regions = self.analysis.planes[self.center_z]
        self.ratios = np.array([(m - 1) / (n - 1) if n > 1 else 0.0
                               for n, m in zip(native_shape, self.shape, strict=True)])
        self.coordinates = np.rint(coordinates * self.ratios).astype(np.int32)
        self.boxes: np.ndarray = np.array([region.bounds for region in regions.values()], dtype=np.float64).reshape(-1, len(self.shape), 2)
        if len(self.boxes):
            low = np.floor((self.boxes[:, :, 0] - 0.5) * self.ratios)
            high = np.ceil((self.boxes[:, :, 1] - 0.5) * self.ratios) + 1
            self.boxes = np.stack((np.maximum(low, 0), np.minimum(high, self.shape)), axis=-1).astype(np.int32)

    @property
    def has_foreground(self) -> bool:
        return bool(len(self.boxes))

    def starts(self, mask: Any, patch_size: tuple[int, ...], rng: np.random.Generator, *,
               foreground: bool, empty: bool, max_padding_ratio: float) -> tuple[int, ...]:
        pads = padding_extents(self.shape, patch_size, max_padding_ratio)
        lower = np.array([-pad[0] for pad in pads])
        upper = np.array([max(n, p) - p - pad[0] for n, p, pad in zip(self.shape, patch_size, pads, strict=True)])
        if not foreground and not empty:
            return tuple(int(rng.integers(a, b + 1)) + pad[0] for a, b, pad in zip(lower, upper, pads, strict=True))
        if not empty and self.has_foreground:
            box = self.boxes[int(rng.integers(len(self.boxes)))]
            complete_low = np.maximum(lower, box[:, 1] - patch_size)
            complete_high = np.minimum(upper, box[:, 0])
            # Reuse foreground locations, not a small fixed set of training crops.
            if self.instances and rng.random() < 0.5 and np.all(complete_low <= complete_high):
                low, high = complete_low, complete_high
            else:
                point = (
                    self.coordinates[int(rng.integers(len(self.coordinates)))] if len(self.coordinates)
                    else (box[:, 0] + box[:, 1] - 1) // 2
                )
                low, high = np.maximum(lower, point - patch_size + 1), np.minimum(upper, point)
            return tuple(int(rng.integers(a, b + 1)) + pad[0] for a, b, pad in zip(low, high, pads, strict=True))
        key = (tuple(patch_size), bool(empty), float(max_padding_ratio))
        if key not in self.pools:
            local = np.random.default_rng(stable_seed(self.seed, 0, repr(key)))
            candidates = []
            for _ in range(128):
                start = np.array([local.integers(a, b + 1) for a, b in zip(lower, upper, strict=True)])
                end = start + patch_size
                intersects = bool(len(self.boxes) and np.any(np.all(
                    (self.boxes[:, :, 0] < end) & (self.boxes[:, :, 1] > start), axis=1
                )))
                if empty and intersects:
                    spatial = tuple(slice(max(0, int(a)), min(n, int(b))) for a, b, n in zip(start, end, self.shape, strict=True))
                    if np.any(mask[spatial]):
                        continue
                candidates.append(tuple(int(a + pad[0]) for a, pad in zip(start, pads, strict=True)))
                if len(candidates) == 16:
                    break
            if not candidates:
                raise DatasetError("No empty crop candidates found for the requested crop size and source")
            self.pools[key] = tuple(candidates)
            while len(self.pools) > 128:
                self.pools.popitem(last=False)
        self.pools.move_to_end(key)
        return self.pools[key][int(rng.integers(len(self.pools[key])))]
