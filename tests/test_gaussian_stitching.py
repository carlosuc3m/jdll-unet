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


@pytest.mark.parametrize("shape,tile", [((13, 17), (8, 8)), ((7, 15, 19), (6, 8, 8)), ((3, 4), (8, 8))])
@pytest.mark.parametrize("blend", ["constant", "gaussian"])
@pytest.mark.parametrize("batch_size", [2, 5])
def test_batched_tiles_preserve_accumulation(shape, tile, blend, batch_size):
    image = np.random.default_rng(2).normal(size=(1, *shape)).astype(np.float32)
    args = (IdentityPrediction(), image, torch.device("cpu"), tile, 0.5)
    serial = tiled_predict(*args, blend_mode=blend)
    batched = tiled_predict(*args, blend_mode=blend, batch_size=batch_size)
    np.testing.assert_array_equal(serial, batched)


def test_batched_low_resolution_outputs_are_resized():
    class LowerResolution(IdentityPrediction):
        def forward(self, image):
            return image[..., ::2, ::2]

    image = np.random.default_rng(3).normal(size=(1, 13, 17)).astype(np.float32)
    args = (LowerResolution(), image, torch.device("cpu"), (8, 8), 0.5)
    np.testing.assert_array_equal(tiled_predict(*args), tiled_predict(*args, batch_size=4))


def test_progress_retains_per_patch_execution():
    events = []

    class Model(IdentityPrediction):
        def forward(self, image):
            assert image.shape[0] == 1
            events.append("predict")
            return image

    class Progress:
        def patch_start(self, starts):
            events.append("start")

        def patch_end(self):
            events.append("end")

    tiled_predict(Model(), np.ones((1, 8, 20), np.float32), torch.device("cpu"), (8, 8),
                  progress=Progress(), batch_size=8)
    assert events == ["start", "predict", "end"] * 3


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_invalid_batch_size_fails(batch_size):
    with pytest.raises(InferenceError, match="batch_size"):
        tiled_predict(IdentityPrediction(), np.ones((1, 8, 8), np.float32), torch.device("cpu"),
                      (8, 8), batch_size=batch_size)


def test_cuda_oom_retries_without_skipping_or_double_counting(monkeypatch):
    calls = []

    class LimitedMemory(IdentityPrediction):
        def forward(self, image):
            calls.append(len(image))
            if len(image) > 2:
                raise torch.cuda.OutOfMemoryError("simulated OOM")
            return image

    # Exercise CUDA recovery on CPU, including in CPU-only CI.
    for name in ("zeros", "ones", "arange"):
        original = getattr(torch, name)

        def cpu_factory(*args, _original=original, **kwargs):
            kwargs["device"] = "cpu"
            return _original(*args, **kwargs)

        monkeypatch.setattr(torch, name, cpu_factory)
    monkeypatch.setattr(torch.Tensor, "to", lambda self, *args, **kwargs: self)
    emptied = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: emptied.append(True))
    image = np.random.default_rng(4).normal(size=(1, 16, 40)).astype(np.float32)
    output = tiled_predict(LimitedMemory(), image, torch.device("cuda"), (8, 8), 0.5,
                           blend_mode="gaussian", batch_size=8)
    np.testing.assert_allclose(output, image, rtol=1e-5, atol=1e-6)
    assert calls[:3] == [8, 4, 2]
    assert max(calls[2:]) <= 2
    assert len(emptied) == 2


def test_single_tile_oom_is_not_swallowed():
    class NoMemory(IdentityPrediction):
        def forward(self, image):
            raise torch.cuda.OutOfMemoryError("simulated OOM")

    with pytest.raises(torch.cuda.OutOfMemoryError):
        tiled_predict(NoMemory(), np.ones((1, 8, 8), np.float32), torch.device("cpu"), (8, 8))


def test_serial_volume_tiles_share_input_storage(monkeypatch):
    image = np.ones((1, 7, 15, 19), np.float32)
    original = torch.from_numpy
    calls = []

    def from_numpy(array):
        assert np.shares_memory(array, image)
        calls.append(array.shape)
        return original(array)

    monkeypatch.setattr(torch, "from_numpy", from_numpy)
    tiled_predict(IdentityPrediction(), image, torch.device("cpu"), (4, 8, 8))
    assert len(calls) > 1
