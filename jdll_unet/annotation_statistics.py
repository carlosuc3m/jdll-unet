"""Reuse component statistics for cleanup, sampling and native-size references."""

from __future__ import annotations

import numpy as np
from numba import njit

from .label_statistics import LabelRegion, MaskAnalysis, _merge


def remap_analysis(
    analysis: MaskAnalysis, mapping: dict[int, int], foreground: np.ndarray | None = None
) -> MaskAnalysis:
    def remap(regions: dict[int, LabelRegion]) -> dict[int, LabelRegion]:
        result: dict[int, LabelRegion] = {}
        for label, region in regions.items():
            target = mapping.get(label, 0)
            if target:
                _merge(result, target, region.count, region.bounds)
        return result

    planes = tuple(remap(plane) for plane in analysis.planes)
    return MaskAnalysis(
        analysis.shape, planes[0] if len(analysis.shape) == 2 else remap(analysis.objects), planes,
        analysis.foreground_indices if foreground is None else foreground,
    )


def largest_components(analysis: MaskAnalysis, source_ids: np.ndarray, binary: bool) -> MaskAnalysis:
    if binary:
        return analysis
    largest: dict[int, int] = {}
    for component, region in analysis.objects.items():
        source = int(source_ids[component])
        previous = largest.get(source)
        if previous is None or region.count > analysis.objects[previous].count:
            largest[source] = component
    return remap_analysis(analysis, {component: component for component in largest.values()})


def reference_domain(analysis: MaskAnalysis, region: tuple) -> MaskAnalysis:
    """Use complete objects within a holdout, never pixels from the other split."""
    if not region:
        return analysis
    shape = tuple(end - start for start, end in region)
    selected = {
        label for label, item in analysis.objects.items()
        if all(lo <= a and b <= hi for (a, b), (lo, hi) in zip(item.bounds, region, strict=True))
    }

    def shift(regions: dict[int, LabelRegion], origin: tuple[int, ...]) -> dict[int, LabelRegion]:
        return {
            label: LabelRegion(item.count, tuple((a - o, b - o) for (a, b), o in zip(item.bounds, origin, strict=True)))
            for label, item in regions.items() if label in selected
        }

    origin = tuple(start for start, _ in region)
    planes = analysis.planes if len(shape) == 2 else analysis.planes[region[0][0]:region[0][1]]
    return MaskAnalysis(shape, shift(analysis.objects, origin), tuple(shift(p, origin[-2:]) for p in planes))


@njit(cache=True)
def component_covariances(labels, count):
    """Optional principal-axis moments in one traversal, without coordinate clouds."""
    means = np.zeros((count + 1, 3), np.float64)
    products = np.zeros((count + 1, 3, 3), np.float64)
    counts = np.zeros(count + 1, np.int64)
    for z in range(labels.shape[0]):
        for y in range(labels.shape[1]):
            for x in range(labels.shape[2]):
                label = labels[z, y, x]
                if label == 0:
                    continue
                coords = (z, y, x)
                counts[label] += 1
                delta = (
                    z - means[label, 0], y - means[label, 1], x - means[label, 2],
                )
                for i in range(3):
                    means[label, i] += delta[i] / counts[label]
                for i in range(3):
                    for j in range(3):
                        products[label, i, j] += delta[i] * (coords[j] - means[label, j])
    for label in range(1, count + 1):
        if counts[label] > 1:
            for i in range(3):
                for j in range(3):
                    products[label, i, j] /= counts[label] - 1
        else:
            products[label] = 0
    return products
