import sys
from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pytest
import tifffile

from jdll_unet import dataset as dataset_module
from jdll_unet.augment import AugmentationConfig, EmptyPatchError, make_augmentation_config, sample_patch
from jdll_unet.config import parse_training_config
from jdll_unet.crop_reading import CropArray
from jdll_unet.crop_sampling import CropSampler
from jdll_unet.dataset import JdllSegmentationDataset
from jdll_unet.errors import ConfigError, DatasetError
from jdll_unet.geometry import DomainReader, inspect_pair
from jdll_unet.image_reading import image_reading_session
from jdll_unet.io import ImageMaskPair, fit_normalization, normalize_image
from jdll_unet.label_statistics import analyze_mask
from jdll_unet.planning import resample_image_mask


def make_pair(root, shape=(8, 40, 48), axes="ZYX", name="sample"):
    root.mkdir(parents=True, exist_ok=True)
    image = np.random.default_rng(10).integers(0, 4096, shape, dtype=np.uint16)
    spatial = tuple(size for axis, size in zip(axes, shape, strict=True) if axis in "ZYX")
    spatial_axes = "".join(axis for axis in axes if axis in "ZYX")
    labels = np.zeros(spatial, dtype=np.uint16)
    labels[..., 12:18, 15:22] = 7
    labels[..., 25:29, 32:36] = 25
    p = ImageMaskPair(root / f"{name}.tif", root / f"{name}_mask.tif", name)
    tifffile.imwrite(p.image, image, metadata={"axes": axes}, photometric="minisblack", compression="deflate")
    tifffile.imwrite(p.mask, labels, metadata={"axes": spatial_axes}, photometric="minisblack", compression="deflate")
    return inspect_pair(p)[0], image, labels


def quiet_config(patch):
    return AugmentationConfig(patch_size=patch, flip_probability=0, rotate90_probability=0,
                              brightness_probability=0, shift_probability=0, contrast_probability=0,
                              gamma_probability=0, noise_probability=0, blur_probability=0)


@pytest.mark.parametrize("dimensions", ["2d", "2.5d", "3d"])
@pytest.mark.parametrize("instance", [False, True])
def test_exact_empty_quota_is_reproducible_and_applies_to_returned_masks(tmp_path, dimensions, instance):
    with image_reading_session():
        shape, axes = ((40, 48), "YX") if dimensions == "2d" else ((8, 40, 48), "ZYX")
        pair, _, _ = make_pair(tmp_path, shape, axes)
        cfg = quiet_config((4, 12, 12) if dimensions == "3d" else (12, 12))
        data = JdllSegmentationDataset([pair], "instance_friendly" if instance else "binary_semantic", None,
            {"type": "minmax"}, cfg, True, dimensions=dimensions, sample_count=20, empty_patch_fraction=0.2)
        outcomes = []
        for epoch in (1, 2, 1):
            data.set_epoch(epoch)
            positives = []
            for index in range(len(data)):
                image, target = data[index]
                mask = target["instances" if instance else "semantic"]
                positive = bool(mask.any())
                assert positive != bool(data.epoch_empty_flags[index])
                assert target["valid"].any()
                assert image.shape[1:] == cfg.patch_size
                positives.append(positive)
            assert positives.count(False) == 4
            outcomes.append(positives)
        assert outcomes[0] == outcomes[2]


@pytest.mark.parametrize("value", [True, -0.1, 1, float("inf"), float("nan")])
def test_invalid_empty_fractions_fail_at_config_parse(tmp_path, value):
    with pytest.raises(ConfigError, match="empty_patch_fraction"):
        parse_training_config({"model_name": "x", "output_dir": tmp_path, "dataset_path": tmp_path, "empty_patch_fraction": value})


def test_quota_rounds_up_and_zero_remains_backward_compatible(tmp_path):
    pair, _, _ = make_pair(tmp_path)
    data = JdllSegmentationDataset([pair], "binary_semantic", None, None, quiet_config((4, 12, 12)), True,
                                   dimensions="3d", sample_count=7, empty_patch_fraction=0.2)
    data.set_epoch(1)
    assert int(data.epoch_empty_flags.sum()) == 2
    cfg = parse_training_config({"model_name": "x", "dataset_path": tmp_path, "output_dir": tmp_path})
    assert cfg.empty_patch_fraction == 0


@pytest.mark.parametrize("axes,shape", [("ZYX", (8, 40, 48)), ("CZYX", (2, 8, 40, 48)),
                                       ("ZYXC", (8, 40, 48, 2)), ("YXC", (40, 48, 3))])
def test_region_read_axes_and_single_z_plane(tmp_path, axes, shape):
    with image_reading_session() as session:
        pair, _, _ = make_pair(tmp_path, shape, axes)
        reader = DomainReader(max_bytes=0, session=session)
        from jdll_unet.geometry import load_domain_image
        whole = load_domain_image(pair, reader=reader, raw=True)
        stats = fit_normalization(whole)
        source = CropArray(pair, reader, statistics=stats)
        spatial = (slice(2, 3), slice(5, 17), slice(9, 23)) if "Z" in axes else (slice(5, 17), slice(9, 23))
        expected = normalize_image(whole, statistics=stats)[(slice(None), *spatial)]
        reader.session.pixels = Mock(side_effect=AssertionError("Unexpected full pixel read"))
        region = Mock(wraps=reader.session.region)
        reader.session.region = region
        np.testing.assert_array_equal(source[(slice(None), *spatial)], expected)
        assert region.call_count == 1
        assert reader.bytes == 0


@pytest.mark.parametrize("spacing,target", [((2, 0.7, 0.7), (1, 1, 1)), ((1, 1, 1), (2, 0.5, 0.5)),
                                            ((1, 1, 1), (20, 1, 1))])
def test_lazy_resampling_matches_full_volume_grid_and_normalization_order(tmp_path, spacing, target):
    with image_reading_session() as session:
        pair, image, mask = make_pair(tmp_path)
        reader = DomainReader(max_bytes=0, session=session)
        stats = fit_normalization(image[None])
        expected, labels = resample_image_mask(normalize_image(image[None], statistics=stats), mask, spacing, target)
        shape = tuple(labels.shape)
        source = CropArray(pair, reader, statistics=stats, resampled_shape=shape)
        target_source = CropArray(pair, reader, mask=True, original_mask=True, resampled_shape=shape)
        for spatial in ((slice(0, 1), slice(3, 13), slice(2, 15)),
                        tuple(slice(max(0, n - 3), n) for n in shape)):
            np.testing.assert_allclose(source[(slice(None), *spatial)], expected[(slice(None), *spatial)], atol=2e-6)
            np.testing.assert_array_equal(target_source[spatial], labels[spatial])


def test_statistics_and_mask_analysis_survive_decoded_cache_eviction(tmp_path, monkeypatch):
    with image_reading_session() as session:
        pair, _, _ = make_pair(tmp_path)
        session.domain_reader = DomainReader(max_bytes=0, session=session)
        fit = Mock(wraps=dataset_module.fit_normalization)
        analyze = Mock(wraps=dataset_module.analyze_mask)
        monkeypatch.setattr(dataset_module, "fit_normalization", fit)
        monkeypatch.setattr(dataset_module, "analyze_mask", analyze)
        data = JdllSegmentationDataset([pair], "binary_semantic", None, None, quiet_config((4, 12, 12)), True,
                                       dimensions="3d", sample_count=10, empty_patch_fraction=0.2)
        for index in range(10):
            data[index]
        assert fit.call_count == 1
        assert analyze.call_count == 1
        assert session.domain_reader.bytes == 0


def test_validation_is_identical_across_epochs_and_preserves_empty_crop(tmp_path):
    with image_reading_session():
        pair, _, _ = make_pair(tmp_path)
        data = JdllSegmentationDataset([pair], "binary_semantic", None, None, quiet_config((4, 4, 4)), False,
                                       dimensions="3d", empty_patch_fraction=0.2)
        first = data[0]
        data.set_epoch(30)
        later = data[0]
        np.testing.assert_array_equal(first[0], later[0])
        np.testing.assert_array_equal(first[1]["semantic"], later[1]["semantic"])
        assert not first[1]["semantic"].any()


def test_stratified_validation_is_fixed_and_keeps_empty_sources(tmp_path):
    with image_reading_session():
        pair, _, _ = make_pair(tmp_path)
        empty, _, mask = make_pair(tmp_path, name="empty")
        tifffile.imwrite(empty.mask, np.zeros_like(mask), metadata={"axes": "ZYX"}, photometric="minisblack")
        empty, _ = inspect_pair(replace(empty, plane_positive_counts=(), label_values=()))
        cfg = quiet_config((4, 8, 8))
        cfg.foreground_probability = 0.5
        data = JdllSegmentationDataset([pair, empty], "binary_semantic", None, None, cfg, False,
                                       dimensions="3d", sample_count=40)
        first = [data[i] for i in range(len(data))]
        data.set_epoch(10)
        later = [data[i] for i in range(len(data))]
        assert any(bool(target["semantic"].any()) for _, target in first)
        assert any(not bool(target["semantic"].any()) for _, target in first[::2])
        assert all(not bool(target["semantic"].any()) for _, target in first[1::2])
        for a, b in zip(first, later, strict=True):
            np.testing.assert_array_equal(a[0], b[0])
            np.testing.assert_array_equal(a[1]["semantic"], b[1]["semantic"])


def test_candidates_are_not_foreground_centered_and_cache_is_bounded():
    mask = np.zeros((80, 80), dtype=np.uint8)
    mask[40, 40] = 1
    sampler = CropSampler(analyze_mask(mask), mask.shape, 3)
    rng = np.random.default_rng(4)
    positions = [sampler.starts(mask, (20, 20), rng, foreground=True, empty=False, max_padding_ratio=1)
                 for _ in range(100)]
    assert len(set(positions)) > 50
    assert all(y <= 40 < y + 20 and x <= 40 < x + 20 for y, x in positions)
    assert any(position != (30, 30) for position in positions)
    sampler.starts(mask, (20, 20), rng, foreground=False, empty=True, max_padding_ratio=1)
    pools = sampler.pools.copy()
    sampler.starts(mask, (20, 20), rng, foreground=False, empty=True, max_padding_ratio=1)
    assert sampler.pools == pools


def test_foreground_reservoir_is_bounded_and_preserves_true_locations():
    mask = np.zeros((50, 80, 90), dtype=np.uint16)
    mask[:, :60] = 7
    stats = analyze_mask(mask)
    assert len(stats.foreground_indices) == 100_000
    assert len(np.unique(stats.foreground_indices)) == 100_000
    assert np.all(mask.ravel()[stats.foreground_indices] == 7)
    assert stats.foreground_indices.dtype == np.uint32


def test_impossible_empty_quota_reports_error_without_changing_masks(tmp_path):
    with image_reading_session():
        pair, _, mask = make_pair(tmp_path)
        mask[:] = 1
        tifffile.imwrite(pair.mask, mask, metadata={"axes": "ZYX"}, photometric="minisblack")
        pair, _ = inspect_pair(replace(pair, plane_positive_counts=(), label_values=()))
        data = JdllSegmentationDataset([pair], "binary_semantic", None, None, quiet_config((4, 12, 12)), True,
                                       dimensions="3d", sample_count=5, empty_patch_fraction=0.2)
        data.set_epoch(1)
        index = int(np.flatnonzero(data.epoch_empty_flags)[0])
        with pytest.raises(DatasetError, match="No empty patch"):
            data[index]
        np.testing.assert_array_equal(tifffile.imread(pair.mask), mask)


@pytest.mark.parametrize("defer", [False, True])
def test_quota_survives_balanced_augmentation_and_instance_scale_jitter(tmp_path, defer):
    with image_reading_session():
        pair, _, _ = make_pair(tmp_path, (16, 60, 64))
        cfg = make_augmentation_config("balanced", (8, 16, 16), True, 0.4,
            {"instance_scale_enabled": True, "target_object_diameter_px": 6})
        data = JdllSegmentationDataset([pair], "instance_friendly", None, None, cfg, True,
            dimensions="3d", sample_count=30, empty_patch_fraction=0.2,
            instance_sizes={pair.stem: 6}, defer_spatial=defer, defer_photometric=defer)
        data.set_epoch(1)
        for index in range(len(data)):
            _, target = data[index]
            assert bool(target["instances"].any()) != bool(data.epoch_empty_flags[index])
            assert ((target["instances"] == 0) & target["valid"].bool()).any()


def test_rejected_mask_crop_does_not_decode_image(tmp_path):
    with image_reading_session() as session:
        pair, _, _ = make_pair(tmp_path)
        reader = DomainReader(max_bytes=0, session=session)
        image = CropArray(pair, reader)
        empty_mask = np.zeros(pair.domain_shape, dtype=np.uint8)
        session.region = Mock(side_effect=AssertionError("Image should not be read for a rejected mask"))
        with pytest.raises(EmptyPatchError):
            sample_patch(image, empty_mask, (4, 12, 12), np.random.default_rng(4),
                         skip_empty=True, include_empty_after_max_retries=False)


@pytest.mark.parametrize("backend", ["missing", "unsupported"])
def test_region_read_falls_back_when_optional_backend_is_unavailable(tmp_path, monkeypatch, backend):
    with image_reading_session() as session:
        pair, raw, _ = make_pair(tmp_path)
        if backend == "missing":
            monkeypatch.setitem(sys.modules, "zarr", None)
        else:
            monkeypatch.setattr(tifffile.TiffPageSeries, "aszarr", Mock(side_effect=NotImplementedError))
        selection = (slice(1, 3), slice(5, 15), slice(9, 19))
        if backend == "unsupported":
            with pytest.warns(RuntimeWarning, match="region backend"):
                result = session.region(pair.image, selection)
        else:
            result = session.region(pair.image, selection)
        np.testing.assert_array_equal(result, raw[selection])
