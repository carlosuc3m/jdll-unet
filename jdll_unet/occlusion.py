"""Bounded, object-relative square occlusion planned without device synchronization."""

from __future__ import annotations

import math

import numpy as np


def plan_cutout(
    validity: np.ndarray,
    diameter: float,
    rng: np.random.Generator,
    *,
    probability: float,
    max_fraction: float,
    diameter_fraction: float,
    max_attempts: int,
) -> np.ndarray:
    """Erase uniformly placed squares up to a sampled unique-valid-pixel budget.

    Overlapping squares count only once. The last square is omitted when it
    would exceed the budget; annotations never influence square placement.
    """
    if validity.ndim != 2:
        raise ValueError("Object-relative cutout currently supports 2D patches only")
    if not math.isfinite(diameter) or diameter <= 0:
        raise ValueError("Cutout requires a finite positive reference diameter")
    erase = np.zeros(validity.shape, dtype=bool)
    if rng.random() >= probability or max_fraction == 0:
        return erase
    budget = math.floor(float(rng.random()) * max_fraction * np.count_nonzero(validity))
    height, width = validity.shape
    side = min(height, width, max(1, math.floor(diameter * diameter_fraction + 0.5)))
    covered = 0
    for _ in range(max_attempts):
        if covered >= budget:
            break
        y, x = int(rng.integers(height - side + 1)), int(rng.integers(width - side + 1))
        view = erase[y:y + side, x:x + side]
        support = validity[y:y + side, x:x + side]
        addition = int(np.count_nonzero(support & ~view))
        if covered + addition > budget:
            break
        if addition:
            view |= support
            covered += addition
    return erase
