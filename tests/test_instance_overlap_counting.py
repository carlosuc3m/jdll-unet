import numpy as np
import pytest

from jdll_unet.metrics import _instance_label_metrics


@pytest.mark.parametrize("shape", [(7, 13), (5, 7, 13), (2, 1025, 1025)])
def test_integer_encoding_preserves_sparse_unsigned_ids_and_chunk_boundaries(shape):
    rng = np.random.default_rng(12)
    truth = rng.integers(0, 4, size=shape, dtype=np.uint64)
    pred = rng.integers(0, 5, size=shape, dtype=np.uint64)
    expected = _instance_label_metrics(pred, truth, threshold=.1)
    # Ordered relabelling preserves deterministic tie-breaking. Values that
    # cannot safely be encoded use the structured-pair fallback.
    sparse_truth = np.array([0, 17, 2**63 + 5, 2**64 - 1], dtype=np.uint64)[truth]
    sparse_pred = np.array([0, 3, 32, 2**63 + 1, 2**64 - 2], dtype=np.uint64)[pred]
    assert _instance_label_metrics(sparse_pred, sparse_truth, threshold=.1) == expected


def test_foreground_overlap_excludes_background_pairs():
    truth = np.array([[0, 7, 7, 0], [0, 0, 0, 19]], dtype=np.uint64)
    pred = np.array([[0, 3, 3, 0], [0, 8, 0, 0]], dtype=np.uint64)
    result = _instance_label_metrics(pred, truth)
    assert result["object_precision"] == .5
    assert result["object_recall"] == .5
    assert result["object_f1"] == .5
    assert result["panoptic_quality"] == .5
    assert result["aggregated_jaccard"] == .5
    assert result["split_error_rate"] == result["merge_error_rate"] == 0


def test_empty_overlap_counts_are_well_defined():
    empty = np.zeros((17, 19), np.uint8)
    assert _instance_label_metrics(empty, empty)["object_f1"] == 1
    assert _instance_label_metrics(np.ones_like(empty), empty)["object_f1"] == 0
