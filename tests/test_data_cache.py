from dataclasses import asdict
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import tifffile

from jdll_unet import geometry
from jdll_unet.augment import AugmentationConfig, make_augmentation_config
from jdll_unet.config import parse_training_config
from jdll_unet.dataset import JdllSegmentationDataset
from jdll_unet.errors import ConfigError
from jdll_unet.geometry import DomainReader, inspect_pair, resolve_data_cache_bytes
from jdll_unet.image_reading import image_reading_session
from jdll_unet.io import ImageMaskPair


@pytest.mark.parametrize("available,expected", [(None, 64), (0, 0), (100 * 1024**2, 10), (20 * 1024**3, 512)])
def test_auto_cache_is_memory_bounded(monkeypatch, available, expected):
    monkeypatch.setattr(geometry, "available_host_memory", lambda: available)
    assert resolve_data_cache_bytes("auto") == expected * 1024**2
    assert resolve_data_cache_bytes(32.5) == int(32.5 * 1024**2)
    assert resolve_data_cache_bytes(0) == 0


def test_available_memory_respects_container_limit(monkeypatch):
    values = {
        "/proc/meminfo": "MemAvailable: 2097152 kB\n",
        "/sys/fs/cgroup/memory.max": str(300 * 1024**2),
        "/sys/fs/cgroup/memory.current": str(100 * 1024**2),
    }

    def read(path, *args, **kwargs):
        if str(path) not in values:
            raise FileNotFoundError(path)
        return values[str(path)]

    monkeypatch.setattr(Path, "read_text", read)
    assert geometry.available_host_memory() == 200 * 1024**2


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), "invalid"])
def test_cache_config_rejects_invalid_budgets(tmp_path, value):
    with pytest.raises(ConfigError, match="data_cache_mb"):
        parse_training_config(
            {"model_name": "x", "dataset_path": tmp_path, "output_dir": tmp_path, "data_cache_mb": value}
        )


def test_planning_and_datasets_share_decoded_pixels(tmp_path):
    image, mask = tmp_path / "image.tif", tmp_path / "mask.tif"
    tifffile.imwrite(image, np.ones((8, 8), np.uint8))
    tifffile.imwrite(mask, np.ones((8, 8), np.uint8))
    with image_reading_session() as session:
        session.domain_reader = DomainReader(max_bytes=128, session=session)
        session.pixels = Mock(wraps=session.pixels)
        pair, _ = inspect_pair(ImageMaskPair(image, mask, "case"))
        for _ in range(2):
            dataset = JdllSegmentationDataset(
                [pair], "binary_semantic", [1], None, AugmentationConfig(patch_size=(8, 8)), False
            )
            assert dataset.reader is session.domain_reader
            dataset._load_item(0)
        assert session.pixels.call_count == 2
        assert session.domain_reader.bytes <= 128


def test_reader_evicts_least_recent_and_can_disable_cache(tmp_path):
    paths = [tmp_path / f"{index}.tif" for index in range(3)]
    for index, path in enumerate(paths):
        tifffile.imwrite(path, np.full((8, 8), index, np.uint8))
    reader = DomainReader(max_bytes=128)
    reader.session.pixels = Mock(wraps=reader.session.pixels)
    for index in (0, 1, 0, 2):
        np.testing.assert_array_equal(reader.read(paths[index]), index)
    assert reader.bytes == 128
    assert reader.session.pixels.call_count == 3
    reader.read(paths[1])
    assert reader.session.pixels.call_count == 4
    disabled = DomainReader(max_bytes=0)
    disabled.read(paths[0])
    assert disabled.bytes == 0 and not disabled.cache


def test_exported_augmentation_cannot_override_new_patch_plan():
    exported = asdict(AugmentationConfig(patch_size=(32, 32)))
    exported.update(patch_size=[32, 32], training_scale_jitter=[0.5, 2.0])
    cfg = make_augmentation_config("balanced", (16, 16), True, 0.5, exported)
    assert cfg.patch_size == (16, 16)
    assert cfg.training_scale_jitter == (0.5, 2.0)
