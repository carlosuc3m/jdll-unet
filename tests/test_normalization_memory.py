import numpy as np
import pytest

from jdll_unet.io import fit_normalization


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
@pytest.mark.parametrize("quantiles", [(0, 100), (1, 99.8), (20.125, 74.6)])
def test_bounded_integer_percentiles_match_linear_numpy(dtype, quantiles, monkeypatch):
    image = np.random.default_rng(2).integers(0, np.iinfo(dtype).max + 1, (2, 13, 25, 42), dtype=dtype)
    image = image.transpose(0, 3, 2, 1)
    expected = [np.percentile(channel.astype(np.float32), quantiles) for channel in image]
    monkeypatch.setattr(np, "percentile", lambda *a, **k: pytest.fail("Integer statistics must not partition a full volume"))
    stats = fit_normalization(image, {"type": "percentile", "low": quantiles[0], "high": quantiles[1]})
    for (offset, scale), (low, high) in zip(stats["channels"], expected, strict=True):
        np.testing.assert_allclose([offset, scale], [low, max(high - low, 1e-6)], rtol=1e-6, atol=1e-7)


def test_constant_integer_and_none_normalization():
    image = np.full((1, 7, 13), 65535, dtype=np.uint16)
    assert fit_normalization(image)["channels"] == [(65535.0, 1e-6)]
    assert fit_normalization(image, {"type": "none"})["channels"] == [(0.0, 1.0)]
