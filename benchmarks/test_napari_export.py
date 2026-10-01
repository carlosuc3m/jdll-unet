import numpy as np
import tifffile
from scipy import ndimage as ndi

from benchmarks.export_napari_example import choose_crop, seed_details, write_volume

SETTINGS = {"threshold": 0.5, "seed_distance_threshold": 0.35, "seed_boundary_threshold": 0.5, "seed_h": 0.1, "min_seed_size": 3}


def test_fallback_for_no_distance_peak():
    foreground = np.zeros((7, 11, 13), dtype=np.float32)
    foreground[1:3, 2:4, 2:5] = 1
    foreground[4:6, 7:9, 8:11] = 1
    learned, fallback, stats = seed_details(foreground, foreground * 0, foreground * 0.1, SETTINGS)
    assert learned.shape == (0, 3)
    np.testing.assert_array_equal(fallback, [[1, 2, 2], [4, 7, 8]])
    assert stats["fallback_seed_components"] == 2


def test_isolated_peak_discarded_by_min_seed_size():
    foreground = np.ones((7, 7, 7), dtype=np.float32)
    distance = np.zeros_like(foreground)
    distance[3, 3, 3] = 0.9
    learned, fallback, stats = seed_details(foreground, foreground * 0, distance, SETTINGS)
    assert stats["candidate_seed_components"] == 1
    assert stats["discarded_small_seed_components"] == 1
    assert learned.shape == (0, 3)
    np.testing.assert_array_equal(fallback, [[3, 3, 3]])
    learned, fallback, stats = seed_details(foreground, foreground * 0, distance, {**SETTINGS, "min_seed_size": 1})
    assert len(learned) == 1 and len(fallback) == 0


def test_plateau_seed_survives():
    foreground = np.ones((7, 7, 7), dtype=np.float32)
    distance = np.zeros_like(foreground)
    distance[2:4, 2:4, 2:4] = 0.9
    learned, fallback, stats = seed_details(foreground, foreground * 0, distance, SETTINGS)
    np.testing.assert_array_equal(learned, [[2.5, 2.5, 2.5]])
    assert not len(fallback)
    assert stats["learned_seed_components"] == 1


def test_crop_contains_good_and_unmatched_instances():
    mask = np.zeros((30, 80, 90), dtype=np.uint8)
    mask[2:5, 12:15, 13:16] = 1
    mask[6:9, 18:21, 19:22] = 2
    crop = choose_crop(mask, {"1": 0.7, "2": 0.0}, (20, 40, 40))
    assert mask[crop].shape == (20, 40, 40)
    assert set(np.unique(mask[crop])) == {0, 1, 2}


def test_tiff_round_trip_keeps_compact_ids_and_axes(tmp_path):
    labels, _ = ndi.label(np.random.default_rng(4).random((9, 12, 13)) > 0.9)
    labels = labels.astype(np.uint16)
    path = tmp_path / "labels.ome.tif"
    write_volume(path, labels, (2.0, 0.208, 0.208))
    with tifffile.TiffFile(path) as source:
        assert source.series[0].axes == "ZYX"
        assert source.series[0].dtype == np.uint16
        np.testing.assert_array_equal(source.asarray(), labels)
        assert 'PhysicalSizeZ="2.0"' in source.ome_metadata
