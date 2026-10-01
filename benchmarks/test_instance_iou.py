import json

import numpy as np
import pytest
from scipy import ndimage as ndi

from benchmarks.evaluate_instance_iou import (
    InstanceOverlap,
    aggregate,
    key,
    native_region,
    prepare_truth,
    reconstruct,
    write_summary,
)
from jdll_unet.planning import restore_continuous_maps
from jdll_unet.postprocess import postprocess_instance

SETTINGS = {
    "method": "distance_boundary_watershed", "threshold": 0.5, "seed_distance_threshold": 0.35,
    "seed_boundary_threshold": 0.5, "seed_h": 0.1, "min_seed_size": 3,
    "min_object_size": 0, "boundary_weight": 1.0, "connectivity": "face",
}


def same_partition(left, right):
    np.testing.assert_array_equal(left == 0, right == 0)
    pairs = np.unique(np.stack([left.ravel(), right.ravel()], axis=1), axis=0)
    assert len(pairs) == len(np.unique(left)) == len(np.unique(right))


def test_perfect_nonconsecutive_ids():
    truth = np.array([[0, 7, 7, 0, 12, 12]], dtype=np.uint8)
    sizes = np.bincount(truth.ravel())
    accumulator = InstanceOverlap(sizes)
    accumulator.update(np.array([[0, 19, 19, 0, 2, 2]]), truth)
    metrics = accumulator.metrics()
    assert metrics["mean_gt_iou"] == 1
    assert metrics["matching_iou_0.5"]["f1"] == 1


def test_missed_objects_and_false_positives():
    truth = np.array([[1, 1, 0, 2, 2, 0, 0]])
    accumulator = InstanceOverlap(np.bincount(truth.ravel()))
    accumulator.update(np.array([[1, 1, 0, 0, 0, 0, 7]]), truth)
    metrics = accumulator.metrics()
    assert metrics["mean_gt_iou"] == 0.5
    assert metrics["matched_only_mean_iou"] == 1
    assert metrics["unmatched_gt"] == 1
    assert metrics["matching_iou_0.5"]["fp"] == 1


def test_merged_objects_match_only_once():
    truth = np.array([[1, 1, 0, 2, 2]])
    accumulator = InstanceOverlap(np.bincount(truth.ravel()))
    accumulator.update(np.ones_like(truth), truth)
    metrics = accumulator.metrics()
    assert metrics["mean_gt_iou"] == 0.2
    assert metrics["unmatched_gt"] == 1


def test_assignment_maximizes_total_iou_not_greedy_best_pair():
    truth = np.array([1] * 10 + [2] * 4)
    prediction = np.array([1] * 6 + [2] * 4 + [1] * 4)
    accumulator = InstanceOverlap(np.bincount(truth))
    accumulator.update(prediction, truth)
    metrics = accumulator.metrics()
    assert metrics["mean_gt_iou"] == pytest.approx(0.4)
    assert metrics["unmatched_gt"] == 0


def test_full_object_union_and_independent_regions():
    accumulator = InstanceOverlap(np.array([0, 10]))
    accumulator.update(np.array([1, 1, 1, 1]), np.array([1, 1, 1, 0]))
    accumulator.update(np.array([1]), np.array([0]))
    metrics = accumulator.metrics()
    assert metrics["mean_gt_iou"] == pytest.approx(3 / 11)
    assert metrics["predicted_instances"] == 2


def test_empty_predictions():
    result = InstanceOverlap(np.array([0, 9, 3])).metrics()
    assert result["mean_gt_iou"] == 0
    assert result["matching_iou_0.5"]["fn"] == 2


def test_summary_exports_inference_overrides(tmp_path):
    truth = np.array([0, 1, 1, 0, 2, 2])
    results = {}
    for threshold, prediction in ((0.5, np.zeros_like(truth)), (0.94, truth)):
        accumulator = InstanceOverlap(np.bincount(truth))
        accumulator.update(prediction, truth)
        metrics = accumulator.metrics()
        settings = {**SETTINGS, "threshold": threshold}
        results[key(settings)] = {"settings": settings, "cases": {"case": metrics}, "aggregate": aggregate([metrics])}
    (tmp_path / "results.json").write_text(json.dumps({"checkpoint": "/model.pt", "results": results}))
    write_summary(tmp_path)
    override = json.loads((tmp_path / "recommended_postprocessing.json").read_text())
    assert override["postprocessing"]["threshold"] == 0.94
    report = json.loads((tmp_path / "results.json").read_text())
    assert report["best_by_pooled_gt_iou"] == "fg=0.94,distance=0.35,boundary=0.5"
    assert "1.000000" in (tmp_path / "README.md").read_text()


def test_native_interpolation():
    compact = np.random.default_rng(7).normal(size=(5, 8, 9)).astype(np.float32)
    shape = (12, 31, 29)
    full = restore_continuous_maps(compact[None], shape)[0]
    for box in ((slice(0, 12), slice(0, 31), slice(0, 29)), (slice(2, 8), slice(4, 14), slice(7, 28))):
        np.testing.assert_allclose(native_region(compact, shape, box), full[box], atol=1e-6)


def test_sparse_repairs_keep_original_ids():
    mask = np.array([[[0, 7, 7, 0, 7, 0, 10, 10]]], dtype=np.uint8)
    repaired, sizes = prepare_truth(mask, 1)
    np.testing.assert_array_equal(repaired, [[[0, 7, 7, 0, 11, 0, 10, 10]]])
    assert sizes[7] == 2 and sizes[11] == 1 and sizes.sum() == 5


def test_seeded_watershed_preserves_touching_object_split():
    z, y, x = np.indices((11, 31, 51))
    distance = np.maximum(
        np.exp(-((z - 5) / 4) ** 2 - ((y - 15) / 8) ** 2 - ((x - 12) / 10) ** 2),
        np.exp(-((z - 5) / 4) ** 2 - ((y - 15) / 8) ** 2 - ((x - 38) / 10) ** 2),
    )
    distance = np.minimum(distance, 0.8).astype(np.float32)
    foreground = np.where(distance > 0.04, 0.99, 0.1).astype(np.float32)
    boundary = np.full(distance.shape, 0.05, dtype=np.float32)
    full = postprocess_instance(foreground, boundary, distance, **SETTINGS)["labels"]
    assert int(full.max()) >= 2
    same_partition(full, reconstruct(foreground, boundary, distance, SETTINGS))


@pytest.mark.parametrize("seed", range(8))
def test_regional_reconstruction_matches_library(seed):
    rng = np.random.default_rng(seed)
    shape = (9, 14, 17)
    foreground = rng.random(shape, dtype=np.float32)
    foreground[:, 5:8] = 0
    foreground[3:6] = 0
    boundary, distance = rng.random((2, *shape), dtype=np.float32)
    settings = {**SETTINGS, "seed_distance_threshold": 0.15 if seed % 2 else 0.9}
    full = postprocess_instance(foreground, boundary, distance, **settings)["labels"]
    support, count = ndi.label(foreground >= 0.3, structure=np.ones((3, 3, 3)))
    regional = np.zeros(shape, dtype=np.uint32)
    offset = 0
    for region_id, tight in enumerate(ndi.find_objects(support, max_label=count), start=1):
        box = tuple(slice(max(0, s.start - 1), min(n, s.stop + 1)) for s, n in zip(tight, shape, strict=True))
        fg = np.where(support[box] == region_id, foreground[box], 0)
        labels = reconstruct(fg, boundary[box], distance[box], settings)
        selected = labels > 0
        regional[box][selected] = labels[selected] + offset
        offset += int(labels.max())
    same_partition(full, regional)
