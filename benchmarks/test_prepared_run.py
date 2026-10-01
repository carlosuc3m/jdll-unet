import numpy as np

from benchmarks.prepared_run import load_analysis, repair_copy, save_analysis
from jdll_unet.label_statistics import analyze_mask
from jdll_unet.scale import canonicalize_instance_volume


def test_roi_repairs_match_full_volume_raster_identity_assignment(tmp_path, monkeypatch):
    import shutil
    from collections import namedtuple
    monkeypatch.setattr(shutil, "disk_usage", lambda _: namedtuple("usage", "total used free")(10**12, 0, 10**12))
    mask = np.zeros((7, 21, 24), dtype=np.uint16)
    mask[1:3, 2:5, 3:6] = 7
    mask[1:3, 13:16, 3:6] = 25
    mask[2, 18, 18] = 25
    mask[5, 18, 19] = 7
    mask[4:6, 4:8, 15:20] = 800
    expected = canonicalize_instance_volume(mask)
    path = tmp_path / "labels.npy"
    counts, parents = repair_copy(mask, analyze_mask(mask), path, expected.repaired_components)
    np.testing.assert_array_equal(np.load(path), expected.labels)
    assert counts == {7: 2, 25: 2, 800: 1}
    assert parents == {801: 25, 802: 7}


def test_analysis_round_trip_without_pickle(tmp_path):
    mask = np.zeros((8, 20, 20), dtype=np.uint8)
    mask[2:5, 4:8, 7:10] = 7
    analysis = analyze_mask(mask)
    path = tmp_path / "analysis.npz"
    save_analysis(path, analysis)
    restored = load_analysis(path)
    assert restored.shape == analysis.shape
    assert restored.objects == analysis.objects
    assert restored.planes == analysis.planes
    np.testing.assert_array_equal(restored.foreground_indices, analysis.foreground_indices)
