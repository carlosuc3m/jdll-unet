from types import SimpleNamespace

import numpy as np
import pytest
import torch

from jdll_unet.errors import InferenceError
from jdll_unet.infer import tiled_predict


class IdentityPrediction(torch.nn.Module):
    config = SimpleNamespace(output_channels=1)

    def forward(self, image):
        return image


@pytest.mark.parametrize("shape,tile", [((13, 17), (8, 8)), ((7, 15, 19), (6, 8, 8)), ((3, 4, 5), (6, 8, 8))])
@pytest.mark.parametrize("overlap", [0, 0.25, 0.5])
def test_gaussian_preserves_predictions_and_covers_edges(shape, tile, overlap):
    image = np.random.default_rng(1).normal(size=(1, *shape)).astype(np.float32)
    prediction = tiled_predict(IdentityPrediction(), image, torch.device("cpu"), tile, overlap,
                               blend_mode="gaussian")
    np.testing.assert_allclose(prediction, image, rtol=1e-5, atol=1e-6)


def test_gaussian_reduces_tile_edge_artifacts():
    class EdgeArtifacts(IdentityPrediction):
        def forward(self, image):
            result = torch.ones_like(image)
            result[..., :2] = 0
            result[..., -2:] = 0
            return result

    image = np.ones((1, 8, 20), np.float32)
    uniform = tiled_predict(EdgeArtifacts(), image, torch.device("cpu"), (8, 8), 0.5)
    gaussian = tiled_predict(EdgeArtifacts(), image, torch.device("cpu"), (8, 8), 0.5, blend_mode="gaussian")
    assert np.mean(abs(gaussian[..., 4:-4] - 1)) < np.mean(abs(uniform[..., 4:-4] - 1))


def test_unknown_blending_fails():
    with pytest.raises(InferenceError, match="tile_blending"):
        tiled_predict(IdentityPrediction(), np.ones((1, 8, 8), np.float32), torch.device("cpu"),
                      (8, 8), blend_mode="invalid")
