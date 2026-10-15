from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from jdll_unet import label_statistics
from jdll_unet.errors import DataFormatError
from jdll_unet.label_statistics import analyze_mask


def reference_reservoir(mask, chunk_pixels):
    """Original batched reservoir algorithm, using NumPy foreground positions."""
    rng = np.random.default_rng(0)
    reservoir = np.empty(0, dtype=np.uint32 if mask.size <= 2**32 else np.uint64)
    seen = 0
    height, width = mask.shape[-2:]
    columns = min(width, chunk_pixels)
    rows = max(1, chunk_pixels // columns)
    for z, plane in enumerate(mask[None] if mask.ndim == 2 else mask):
        for y in range(0, height, rows):
            for x in range(0, width, columns):
                tile = plane[y : y + rows, x : x + columns]
                positions = np.flatnonzero(tile)
                count = len(positions)
                if not count:
                    continue
                keep = min(100_000, seen + count)
                take_old = int(rng.hypergeometric(seen, count, keep)) if seen else 0
                old = reservoir[rng.choice(len(reservoir), take_old, replace=False)] if take_old else reservoir[:0]
                selected = positions[rng.choice(count, keep - take_old, replace=False)]
                flat = (z * height + y + selected // tile.shape[1]) * width + x + selected % tile.shape[1]
                reservoir = np.concatenate((old, flat.astype(reservoir.dtype)))
                seen += count
    return reservoir


@pytest.mark.parametrize("dtype", [np.bool_, np.uint8, np.uint16, np.uint32, np.uint64,
                                   np.int8, np.int16, np.int32, np.int64, ">u2", np.float32])
@pytest.mark.parametrize("layout", ["c", "strided", "readonly"])
def test_integer_scan_matches_reference_statistics_and_positions(dtype, layout, monkeypatch):
    monkeypatch.setattr(label_statistics, "STATISTICS_CHUNK_PIXELS", 41)
    mask = np.random.default_rng(17).integers(0, 8, size=(3, 13, 19)).astype(dtype)
    if layout == "strided":
        mask = mask.T[::-1, ::-1]
    if layout == "readonly":
        mask.flags.writeable = False
    analysis = analyze_mask(mask)
    for label, region in analysis.objects.items():
        coordinates = np.argwhere(mask == label)
        assert region.count == len(coordinates)
        assert region.bounds == tuple(zip(coordinates.min(axis=0), coordinates.max(axis=0) + 1, strict=True))
    for plane, regions in zip(mask, analysis.planes, strict=True):
        for label, region in regions.items():
            coordinates = np.argwhere(plane == label)
            assert region.count == len(coordinates)
            assert region.bounds == tuple(zip(coordinates.min(axis=0), coordinates.max(axis=0) + 1, strict=True))
    np.testing.assert_array_equal(analysis.foreground_indices, reference_reservoir(mask, 41))


@pytest.mark.parametrize("shape,chunk", [((600, 900), 1024**2), ((5, 256, 400), 30000)])
def test_reservoir_over_capacity_is_bit_identical(shape, chunk, monkeypatch):
    monkeypatch.setattr(label_statistics, "STATISTICS_CHUNK_PIXELS", chunk)
    mask = np.random.default_rng(9).integers(0, 5, size=shape, dtype=np.uint16)
    result = analyze_mask(mask)
    assert len(result.foreground_indices) == 100_000
    np.testing.assert_array_equal(result.foreground_indices, reference_reservoir(mask, chunk))


def test_integer_fast_path_does_not_repeat_numpy_scans(monkeypatch):
    mask = np.zeros((13, 17), dtype=np.uint16)
    mask[1:3, 5:8] = 7
    mask[8:11, 2:4] = 250
    monkeypatch.setattr(label_statistics.ndi, "find_objects", Mock(side_effect=AssertionError("Repeated scan")))
    monkeypatch.setattr(label_statistics, "validate_mask_labels", Mock(side_effect=AssertionError("Repeated validation")))
    result = analyze_mask(mask)
    assert result.labels == (7, 250)
    assert result.objects[7].count == result.objects[250].count == 6


def test_sparse_fallback_and_dense_next_tile_do_not_leak_partial_counts(monkeypatch):
    monkeypatch.setattr(label_statistics, "STATISTICS_CHUNK_PIXELS", 16)
    mask = np.zeros((6, 4), np.uint64)
    mask[0, :2] = 7
    mask[1, 1] = 2**63 - 1
    mask[4, :3] = 65536
    mask[5, 0] = 7
    result = analyze_mask(mask)
    assert result.labels == (7, 65536, 2**63 - 1)
    assert result.objects[7].count == 3
    assert result.objects[65536].count == 3
    assert result.objects[2**63 - 1].count == 1
    np.testing.assert_array_equal(result.foreground_indices, reference_reservoir(mask, 16))


@pytest.mark.parametrize("dtype,value", [(np.int32, -1), (np.uint64, 2**64 - 1),
                                       (np.float32, 1.5), (np.float64, np.inf), (np.complex64, 1j)])
def test_invalid_label_reports_source_path(dtype, value):
    mask = np.zeros((5, 7), dtype)
    mask[0, 0] = 1
    mask[-1, -1] = value
    with pytest.raises(DataFormatError, match="broken.tif"):
        analyze_mask(mask, Path("broken.tif"))


def test_readonly_memmap_is_not_modified_or_copied_in_full(tmp_path, monkeypatch):
    monkeypatch.setattr(label_statistics, "STATISTICS_CHUNK_PIXELS", 31)
    path = tmp_path / "labels.npy"
    original = np.random.default_rng(8).integers(0, 7, size=(3, 13, 19), dtype=np.uint16)
    np.save(path, original)
    mask = np.load(path, mmap_mode="r")
    result = analyze_mask(mask)
    assert result.objects == analyze_mask(original).objects
    np.testing.assert_array_equal(mask, original)
    np.testing.assert_array_equal(result.foreground_indices, reference_reservoir(original, 31))
