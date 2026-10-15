import numpy as np
import pytest
from skimage.morphology import h_maxima

from jdll_unet import postprocess


@pytest.mark.parametrize("h", [.001, .025, .1, .5, 1.])
@pytest.mark.parametrize("kind", ["empty", "constant", "dense", "sparse", "diagonal", "nested"])
def test_component_maxima_match_global_morphology(kind, h):
    rng = np.random.default_rng(45)
    image = rng.random((96, 81), dtype=np.float32)
    if kind == "empty":
        image[:] = 0
    elif kind == "constant":
        image[:] = .5
    elif kind == "sparse":
        image[rng.random(image.shape) < .8] = 0
    elif kind == "diagonal":
        image[:] = 0
        np.fill_diagonal(image, .5)
        image[20, 20] = .8
    elif kind == "nested":
        image[:] = 0
        for offset in range(0, 40, 3):
            image[offset:-offset or None, offset] = .5
            image[offset:-offset or None, -offset - 1] = .5
            image[offset, offset:-offset or None] = .5
            image[-offset - 1, offset:-offset or None] = .5
    np.testing.assert_array_equal(postprocess._foreground_h_maxima(image, h), h_maxima(image, h=h))


@pytest.mark.parametrize("shape", [(1, 1), (1, 83), (83, 1), (7, 15, 19)])
def test_small_or_volumetric_maps_keep_reference_behavior(shape):
    image = np.random.default_rng(1).random(shape, dtype=np.float32)
    np.testing.assert_array_equal(postprocess._foreground_h_maxima(image, .1), h_maxima(image, h=.1))


def test_reconstruction_is_identical(monkeypatch):
    rng = np.random.default_rng(5)
    foreground = rng.random((96, 81), dtype=np.float32)
    boundary = rng.random((96, 81), dtype=np.float32)
    distance = rng.random((96, 81), dtype=np.float32)
    options = {"min_seed_size": 1, "min_object_size": 4, "seed_distance_threshold": .1}
    actual = postprocess.postprocess_instance(foreground, boundary, distance, **options)
    monkeypatch.setattr(postprocess, "_foreground_h_maxima", lambda image, h: h_maxima(image, h=h))
    expected = postprocess.postprocess_instance(foreground, boundary, distance, **options)
    for key in actual:
        np.testing.assert_array_equal(actual[key], expected[key])
