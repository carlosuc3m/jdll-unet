import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import tifffile
import torch
from PIL import Image

from jdll_unet import geometry
from jdll_unet.augment import AugmentationConfig, make_augmentation_config
from jdll_unet.config import parse_training_config
from jdll_unet.dataset import JdllSegmentationDataset
from jdll_unet.errors import ConfigError
from jdll_unet.geometry import DomainReader, inspect_pair, resolve_data_cache_bytes
from jdll_unet.image_reading import ImageReadSession, image_reading_session
from jdll_unet.io import ImageMaskPair
from jdll_unet.prefetch import BatchPrefetcher


@pytest.mark.parametrize("available,expected", [(None, 64), (0, 0), (100 * 1024**2, 10),
                                              (10 * 1024**3, 1024), (20 * 1024**3, 2048), (64 * 1024**3, 2048)])
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


def test_cached_crops_do_not_resolve_paths_and_detect_symlink_retarget(tmp_path, monkeypatch):
    paths = [tmp_path / f"source_{index}.tif" for index in range(2)]
    for index, path in enumerate(paths):
        tifffile.imwrite(path, np.full((8, 8), index, np.uint8))
    link = tmp_path / "linked.tif"
    link.symlink_to(paths[0])
    reader = DomainReader(max_bytes=256)
    reader.read(link)
    with monkeypatch.context() as patcher:
        patcher.setattr(Path, "resolve", Mock(side_effect=AssertionError("unexpected path resolution")))
        np.testing.assert_array_equal(reader.read(link, selection=(slice(2, 5), slice(1, 4))), 0)
    link.unlink()
    link.symlink_to(paths[1])
    np.testing.assert_array_equal(reader.read(link), 1)


def test_cache_detects_atomic_replacement_with_preserved_mtime(tmp_path):
    import os

    path = tmp_path / "image.tif"
    replacement = tmp_path / "replacement.tif"
    tifffile.imwrite(path, np.zeros((8, 8), np.uint8))
    reader = DomainReader(max_bytes=256)
    reader.read(path)
    before = path.stat()
    tifffile.imwrite(replacement, np.ones((8, 8), np.uint8))
    os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
    replacement.replace(path)
    np.testing.assert_array_equal(reader.read(path), 1)


def png_source(tmp_path, name, value=1, shape=(8, 8)):
    path = tmp_path / f"{name}.png"
    Image.fromarray(np.full(shape, value, np.uint8)).save(path)
    return path


def test_patch_full_decode_is_cached_once_without_a_second_decode(tmp_path, monkeypatch):
    from jdll_unet import image_reading

    path = png_source(tmp_path, "source", value=7)
    decoded = Mock(wraps=image_reading.decode_pixels)
    monkeypatch.setattr(image_reading, "decode_pixels", decoded)
    reader = DomainReader(max_bytes=64)
    for _ in range(3):
        np.testing.assert_array_equal(reader.read(path, selection=(slice(2, 5), slice(1, 4))), 7)
    assert decoded.call_count == 1
    assert reader.bytes == 64
    assert not next(iter(reader.cache.values())).flags.writeable


def test_training_admission_keeps_sources_with_more_planned_requests(tmp_path):
    paths = [png_source(tmp_path, str(i), value=i) for i in range(4)]
    plan = {(os.path.abspath(path), "image"): count for path, count in zip(paths, (10, 2, 1, 20), strict=True)}
    reader = DomainReader(max_bytes=128)
    region = (slice(1, 4), slice(1, 4))
    with reader.cache_plan(plan):
        for i in (0, 1, 2, 3):
            np.testing.assert_array_equal(reader.read(paths[i], selection=region), i)
        assert {key[0] for key in reader.cache} == {str(paths[0]), str(paths[3])}
        assert reader.bytes == 128
        reader.session.region = Mock(wraps=reader.session.region)
        reader.read(paths[0], selection=region)
        reader.session.region.assert_not_called()
    # Validation/other reads retain LRU and are not blocked by the training priority.
    reader.read(paths[2], selection=region)
    assert str(paths[2]) in {key[0] for key in reader.cache}


def test_training_priority_accounts_for_bytes_and_rejects_without_partial_eviction(tmp_path):
    small = png_source(tmp_path, "small")
    large = png_source(tmp_path, "large", shape=(16, 16))
    reader = DomainReader(max_bytes=256)
    plan = {(str(small), "image"): 8, (str(large), "image"): 16}
    with reader.cache_plan(plan):
        reader.read(small)
        reader.read(large)
    assert {key[0] for key in reader.cache} == {str(small)}

    other = png_source(tmp_path, "other")
    medium = png_source(tmp_path, "medium", shape=(8, 16))
    reader = DomainReader(max_bytes=128)
    plan = {(str(small), "image"): 1, (str(other), "image"): 100, (str(medium), "image"): 10}
    with reader.cache_plan(plan):
        reader.read(small)
        reader.read(other)
        before = list(reader.cache)
        reader.read(medium)
    assert list(reader.cache) == before
    assert reader.bytes == 128


def test_priorities_change_with_epoch_and_equal_priorities_use_lru(tmp_path):
    first, second, third = [png_source(tmp_path, str(i)) for i in range(3)]
    reader = DomainReader(max_bytes=64)
    with reader.cache_plan({(str(first), "image"): 10, (str(second), "image"): 1}):
        reader.read(first)
        reader.read(second)
    assert next(iter(reader.cache))[0] == str(first)
    with reader.cache_plan({(str(first), "image"): 1, (str(second), "image"): 10}):
        reader.read(second)
    assert next(iter(reader.cache))[0] == str(second)

    reader = DomainReader(max_bytes=128)
    with reader.cache_plan({(str(path), "image"): 5 for path in (first, second, third)}):
        for path in (first, second, first, third):
            reader.read(path)
    assert {key[0] for key in reader.cache} == {str(first), str(third)}


@pytest.mark.parametrize("kind", ["mapped_tiff", "compressed_tiff", "numpy"])
def test_region_capable_sources_are_not_fully_decoded_to_populate_cache(tmp_path, monkeypatch, kind):
    from jdll_unet import image_reading

    if kind == "compressed_tiff":
        pytest.importorskip("zarr")
    data = np.arange(128 * 160, dtype=np.uint16).reshape(128, 160)
    path = tmp_path / ("source.npy" if kind == "numpy" else "source.tif")
    if kind == "numpy":
        np.save(path, data)
    else:
        tifffile.imwrite(path, data, photometric="minisblack",
                         **({"tile": (32, 32), "compression": "deflate"} if kind == "compressed_tiff" else {}))
    monkeypatch.setattr(image_reading, "decode_pixels", Mock(side_effect=AssertionError("Full decode")))
    reader = DomainReader(max_bytes=data.nbytes * 2)
    with reader.cache_plan({(str(path), "image"): 100}):
        region = (slice(12, 35), slice(10, 42))
        np.testing.assert_array_equal(reader.read(path, selection=region), data[region])
    assert reader.bytes == 0 and not reader.cache


def test_disabled_or_oversized_sources_bypass_admission(tmp_path):
    path = png_source(tmp_path, "source")
    for budget in (0, 32):
        reader = DomainReader(max_bytes=budget)
        with reader.cache_plan({(str(path), "image"): 100}):
            reader.read(path, selection=(slice(1, 4), slice(1, 4)))
        assert reader.bytes == 0 and not reader.cache


def test_epoch_cache_counts_match_the_actual_source_and_plane_plan(tmp_path):
    with image_reading_session() as session:
        pairs = []
        for i, shape in enumerate(((16, 16), (4, 16, 16))):
            paths = [tmp_path / f"{role}{i}.tif" for role in ("image", "mask")]
            for path in paths:
                tifffile.imwrite(path, np.ones(shape, np.uint8), photometric="minisblack",
                                 metadata={"axes": "YX" if len(shape) == 2 else "ZYX"})
            pair, _ = inspect_pair(ImageMaskPair(*paths, str(i)))
            pairs.append(pair)
        dataset = JdllSegmentationDataset(pairs, "binary_semantic", [1], None,
                AugmentationConfig(patch_size=(8, 8)), True, sample_count=100)
        assert dataset.reader is session.domain_reader
        for epoch in (1, 2):
            dataset.set_epoch(epoch)
            counts = np.bincount([dataset.items[int(i)][0] for i in dataset.epoch_indices], minlength=len(pairs))
            for pair, count in zip(pairs, counts, strict=True):
                assert dataset._cache_requests[(str(pair.image), "image")] == count
                assert dataset._cache_requests[(str(pair.mask), "mask")] == count
            assert sum(dataset._cache_requests.values()) == 2 * len(dataset)


@pytest.mark.parametrize("prefetch", [False, True])
def test_cache_policy_does_not_change_sample_coordinates_or_augmented_values(tmp_path, prefetch):
    paths = [tmp_path / f"{role}.tif" for role in ("image", "mask")]
    image = np.arange(16 * 16, dtype=np.uint16).reshape(16, 16)
    mask = np.zeros_like(image)
    mask[3:13, 3:13] = 1
    for path, data in zip(paths, (image, mask), strict=True):
        tifffile.imwrite(path, data, photometric="minisblack")
    def sample(budget):
        with image_reading_session() as session:
            session.domain_reader = DomainReader(max_bytes=budget, session=session)
            pair, _ = inspect_pair(ImageMaskPair(*paths, "case"))
            dataset = JdllSegmentationDataset([pair], "binary_semantic", [1], None,
                    make_augmentation_config("balanced", (8, 8), True, 0.5), True, seed=7, sample_count=8)
            dataset.set_epoch(3)
            plan = dataset.epoch_indices.copy()
            if prefetch:
                loader = BatchPrefetcher(torch.utils.data.DataLoader(dataset, batch_size=None), read_session=session)
                try:
                    samples = list(loader)
                finally:
                    loader.close()
            else:
                samples = [dataset[i] for i in range(len(dataset))]
            return plan, samples
    uncached_plan, uncached = sample(0)
    cached_plan, cached = sample(1024**2)
    np.testing.assert_array_equal(uncached_plan, cached_plan)
    for (left_image, left_target), (right_image, right_target) in zip(uncached, cached, strict=True):
        assert torch.equal(left_image, right_image)
        assert left_target.keys() == right_target.keys()
        assert all(torch.equal(left_target[key], right_target[key]) for key in left_target)


def test_scoped_plans_are_restored_and_reader_remains_worker_pickleable(tmp_path):
    path = png_source(tmp_path, "source")
    reader = DomainReader(max_bytes=64, session=ImageReadSession())
    plan = {(str(path), "image"): 10}
    with reader.cache_plan(plan):
        with pytest.raises(RuntimeError), reader.cache_plan(None):
            raise RuntimeError("test")
        assert geometry._active_cache_plan.get() == (reader, plan)
    assert geometry._active_cache_plan.get() is None
    reader.read(path)
    restored = pickle.loads(pickle.dumps(reader))
    np.testing.assert_array_equal(restored.read(path), 1)
    assert restored.bytes == 64
    assert not next(iter(restored.cache.values())).flags.writeable


def test_concurrent_admission_keeps_byte_accounting_bounded(tmp_path):
    paths = [png_source(tmp_path, str(i), value=i) for i in range(8)]
    reader = DomainReader(max_bytes=128)
    plan = {(str(path), "image"): i + 1 for i, path in enumerate(paths)}
    def read(i):
        with reader.cache_plan(plan):
            np.testing.assert_array_equal(reader.read(paths[i % 8], selection=(slice(1, 4), slice(1, 4))), i % 8)
    with ThreadPoolExecutor(max_workers=4) as workers:
        list(workers.map(read, range(64)))
    assert reader.bytes == sum(array.nbytes for array in reader.cache.values()) <= 128


def test_exported_augmentation_cannot_override_new_patch_plan():
    exported = asdict(AugmentationConfig(patch_size=(32, 32)))
    exported.update(patch_size=[32, 32], training_scale_jitter=[0.5, 2.0])
    cfg = make_augmentation_config("balanced", (16, 16), True, 0.5, exported)
    assert cfg.patch_size == (16, 16)
    assert cfg.training_scale_jitter == (0.5, 2.0)
