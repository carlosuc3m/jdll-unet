import numpy as np
import pytest
import torch
import torch.nn.functional as F

from benchmarks.evaluate_gaussian_instances import foreground_regions, region_probabilities, scaled_image, setting_key
from benchmarks.evaluate_instance_iou import reconstruct
from jdll_unet.planning import restore_continuous_maps
from jdll_unet.postprocess import postprocess_instance


@pytest.mark.parametrize("shape", [(5, 13, 11), (12, 19, 25)])
def test_streaming_scale_matches_full_inference(shape):
    source = np.random.default_rng(9).random((1, 7, 17, 19), dtype=np.float32)
    expected = F.interpolate(torch.from_numpy(source[None]), size=shape, mode="trilinear", align_corners=False)[0].numpy()
    actual = scaled_image(source, shape, lambda *args, **kwargs: None)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_occupancy_groups_every_native_foreground_without_sampling():
    rng = np.random.default_rng(9)
    logits = rng.normal(-2, 1, size=(3, 7, 13, 17)).astype(np.float32)
    shape = (13, 37, 49)
    components, boxes = foreground_regions(logits, shape, 0.2)
    reconstructed = np.zeros(shape, np.float32)
    for region_id, box in enumerate(boxes, 1):
        probabilities = region_probabilities(logits, shape, box, components, region_id)
        reconstructed[box] += probabilities[0]
    expected = 1 / (1 + np.exp(-restore_continuous_maps(logits[:1], shape)[0]))
    np.testing.assert_allclose(reconstructed[expected >= 0.2], expected[expected >= 0.2], rtol=1e-5)
    assert not np.any((reconstructed >= 0.2) & (expected < 0.2))


def test_grouped_postprocessing_matches_full_volume_with_single_voxel_seeds():
    z, y, x = np.indices((7, 19, 35))
    distance = np.maximum(np.exp(-((x - 8) ** 2 + (y - 9) ** 2 + (z - 3) ** 2) / 15),
                          np.exp(-((x - 27) ** 2 + (y - 9) ** 2 + (z - 3) ** 2) / 15)).astype(np.float32)
    probabilities = np.stack((np.where(distance > 0.1, 0.99, 0.01), np.full_like(distance, 0.05), distance))
    probabilities = np.clip(probabilities, 1e-6, 1 - 1e-6)
    logits = np.log(probabilities / (1 - probabilities)).astype(np.float32)
    shape = distance.shape
    settings = {"threshold": 0.5, "seed_distance_threshold": 0.2, "seed_boundary_threshold": 0.5,
                "seed_h": 0.1, "min_seed_size": 1, "min_object_size": 0,
                "connectivity": "face", "method": "distance_boundary_watershed"}
    full = postprocess_instance(*probabilities, **settings)["labels"]
    components, boxes = foreground_regions(logits, shape, 0.2, block=(2, 4, 4))
    regional = np.zeros(shape, np.uint32)
    offset = 0
    for region_id, box in enumerate(boxes, 1):
        probs = region_probabilities(logits, shape, box, components, region_id, block=(2, 4, 4))
        labels = reconstruct(*probs, settings)
        positive = labels > 0
        regional[box][positive] = labels[positive] + offset
        offset += int(labels.max())
    np.testing.assert_array_equal(full == 0, regional == 0)
    pairs = np.unique(np.stack((full.ravel(), regional.ravel()), axis=1), axis=0)
    assert len(pairs) == len(np.unique(full)) == len(np.unique(regional))


def test_cache_key_distinguishes_seed_prominence_and_size():
    assert setting_key({"seed_h": 0.1, "min_seed_size": 1}) != setting_key({"seed_h": 0.2, "min_seed_size": 1})
    assert setting_key({"seed_h": 0.1, "min_seed_size": 1}) != setting_key({"seed_h": 0.1, "min_seed_size": 3})
