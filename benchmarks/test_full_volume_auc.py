import numpy as np
import pytest
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score

from benchmarks.evaluate_full_volume_auc import ProbabilityHistogram


@pytest.mark.parametrize("seed", range(5))
def test_histogram_auc_and_quantization_bound(seed):
    rng = np.random.default_rng(seed)
    scores = rng.random(500).astype(np.float32)
    truth = rng.random(500) < 0.1
    hist = ProbabilityHistogram(16)
    for start in range(0, len(scores), 23):
        hist.update(scores[start : start + 23], truth[start : start + 23])
    result = hist.metrics()
    quantized = np.minimum((scores * 16).astype(np.int64), 15)
    exact = roc_auc_score(truth, scores)
    assert result["roc_auc"] == pytest.approx(roc_auc_score(truth, quantized))
    assert abs(result["roc_auc"] - exact) <= result["roc_auc_error_bound"] + 1e-12
    assert result["average_precision"] == pytest.approx(average_precision_score(truth, quantized))
    precision, recall, _ = precision_recall_curve(truth, quantized)
    assert result["pr_auc_trapezoidal"] == pytest.approx(auc(recall, precision))
    assert result["positive_voxels"] == int(truth.sum())
    assert result["negative_voxels"] == int((~truth).sum())


def test_ties_endpoints_and_threshold():
    scores = np.array([0, 0.5, 0.5, 1], dtype=np.float32)
    truth = np.array([False, False, True, True])
    hist = ProbabilityHistogram(1024)
    hist.update(scores, truth)
    result = hist.metrics()
    assert result["roc_auc"] == pytest.approx(0.875)
    assert result["dice_at_0_5"] == pytest.approx(0.8)
    assert result["recall_at_0_5"] == 1.0


def test_single_class_is_not_reported_as_perfect_auc():
    hist = ProbabilityHistogram(16)
    hist.update(np.zeros(10), np.zeros(10, dtype=bool))
    assert hist.metrics()["roc_auc"] is None


@pytest.mark.parametrize("scores", [np.array([np.nan]), np.array([-0.1]), np.array([1.1])])
def test_invalid_probabilities_rejected(scores):
    with pytest.raises(ValueError):
        ProbabilityHistogram(16).update(scores, np.array([True]))


def test_shape_mismatch_rejected():
    with pytest.raises(ValueError):
        ProbabilityHistogram(16).update(np.zeros(2), np.zeros(3))
