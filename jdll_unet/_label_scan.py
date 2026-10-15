"""Bounded-workspace integer label scan; imported only when analyzing masks."""

import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def scan_integer_tile(tile: np.ndarray, positions: np.ndarray, counts: np.ndarray, bounds: np.ndarray) -> tuple[int, int]:
    """Return foreground length and maximum ID, or (-1, -1) for the generic path."""
    counts[:] = 0
    length = 0
    maximum = 0
    height, width = tile.shape
    for y in range(height):
        for x in range(width):
            value = tile[y, x]
            if value == 0:
                continue
            if value < 0 or value >= len(counts):
                return -1, -1
            # A signed index avoids uint64/int64 promotion to float in max().
            label = np.intp(value)
            positions[length] = y * width + x
            length += 1
            if counts[label] == 0:
                bounds[label, 0] = y
                bounds[label, 1] = y + 1
                bounds[label, 2] = x
                bounds[label, 3] = x + 1
            else:
                bounds[label, 0] = min(bounds[label, 0], y)
                bounds[label, 1] = y + 1
                bounds[label, 2] = min(bounds[label, 2], x)
                bounds[label, 3] = max(bounds[label, 3], x + 1)
            counts[label] += 1
            maximum = max(maximum, int(label))
    return length, maximum
